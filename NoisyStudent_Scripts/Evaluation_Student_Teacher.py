import copy
import torch
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
import os
import pandas as pd
import timeit
import random
from ultralytics import YOLO
import csv
from datetime import datetime



def evaluate(input_teacher, input_student, input_evalSet, input_folderOut):
    IoU_thresh_1 = 0.5

    # Create a global table containing all important information
    header_1 = ['img_name', 'mode', 'class',   'x_min', 'y_min', 'x_max', 'y_max']
    current_time = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

    # In each run, create a new folder for the evaluation results
    folder_name = "Evaluation_" + current_time
    input_folderOut = input_folderOut +"/" +folder_name + "/"
    os.mkdir(input_folderOut)
    os.mkdir(input_folderOut + "EmptyTargetFrames/")
    os.mkdir(input_folderOut + "ResultFrames/")

    file_path_1 = input_folderOut + '/Eval_AnnotsAndDetections_' + current_time + ' .csv'

    with open(file_path_1, 'a', newline='') as file1:
        # Create a CSV writer
        writer_Detections = csv.writer(file1)
        # Write the header row
        writer_Detections.writerow(header_1)

    # Create the output csv
    header = ['file', 'TP', 'FP', 'FN']
    file_path = input_folderOut + '/Eval_Results.csv' + current_time + ' .csv'
    with open(file_path, 'a', newline='') as file:
        # Create a CSV writer
        writer = csv.writer(file)
        # Write the header row
        writer.writerow(header)

    # Load the models
    print("Start model evaluation")
    model_teacher = YOLO(input_teacher)
    print("Loading teacher-model done.")
    model_student = YOLO(input_student)
    print("Loading student-model done.")
    print('\n The folderpath is: %s/%s \n  ' %((input_evalSet),(folder_name)))
    #print(torch.cuda.current_device())

    # Parameter
    target_radius = 160
    target_radius2 = 540
    font = cv2.FONT_HERSHEY_SIMPLEX
    classes_list = ["head", "back", "piglet", "tail"]
    # Check the files
    all_files_image_folder = os.listdir(input_evalSet + "/labels")

    # Loop through the annotations
    # i = "Kamera2-20211115-170841-1636992521_0_560_frame6149.txt"
    # i = "Kamera-00220211011-151632-1633958192_190_560_frame120.txt"
    for i in all_files_image_folder:
        #i = "Kamera2-20211115-170841-1636992521_0_560_frame1118.txt"
        #print(i)
        ## Image specific variables
        tail_piglet_TP_thresh_1 = []
        tail_piglet_FN_thresh_1 = []
        tail_piglet_FP_thresh_1 = []

        name_only = i.split(".")[0]
        #print(name_only)
        ## Load the image
        img_orig = cv2.imread(input_evalSet + "images/" + name_only + ".png", 1 )
        if img_orig.shape != (1080,1920,3):
            img_orig = cv2.resize(img_orig, (1920, 1080))
        img_dim = img_orig.shape
        ## load the annotation
        annot = pd.read_csv(input_evalSet + "labels/" + name_only + ".txt", delimiter=' ', header = None)
        annot_colNames = ["class_int","x_center_rel","y_center_rel", "bw_rel", "bh_rel"]
        annot.columns = annot_colNames
        annot["x_center"] = annot["x_center_rel"] * img_dim[1]
        annot["y_center"] = annot["y_center_rel"] * img_dim[0]
        annot["x_center"] = annot["x_center"].astype(int)
        annot["y_center"] = annot["y_center"].astype(int)
        annot["bw"] = annot["bw_rel"] * img_dim[1] / 2
        annot["bh"] = annot["bh_rel"] * img_dim[0]  / 2
        annot["bw"] = annot["bw"].astype(int)
        annot["bh"] = annot["bh"].astype(int)
        class_map = {0: 'head', 1: 'back', 2: 'piglet', 3:"tail"}
        annot['class'] = annot['class_int'].apply(lambda x: class_map.get(x, np.nan))

        # Compute target area
        back_center_x = int(annot[annot["class"].str.contains("back")]["x_center"].iloc[0])
        back_center_y = int(annot[annot["class"].str.contains("back")]["y_center"].iloc[0])
        empty_frame = np.zeros(shape=(img_dim[0], img_dim[1], 1))
        target_mask = cv2.circle(empty_frame, (back_center_x, back_center_y), target_radius, (1), -1)
        ## Crop target area for student model
        frame_copy = copy.deepcopy(img_orig)
        inds = np.where(target_mask == 0)
        frame_copy[inds[0], inds[1]] = 0
        target_rect_ymin, target_rect_xmin, target_rect_ymax, target_rect_xmax = int(
            back_center_y - target_radius), int(back_center_x - target_radius), int(back_center_y + target_radius), int(
            back_center_x + target_radius)
        target_crop = frame_copy[target_rect_ymin:target_rect_ymax, target_rect_xmin: target_rect_xmax]
        target_crop = cv2.resize(target_crop, (img_dim[0],img_dim[0]))
        # save the empty target_crop image
        cv2.imwrite(input_folderOut+ "EmptyTargetFrames/" + name_only + ".png", target_crop)
        target_crop_dim = target_crop.shape
        # Create the annotation for the target reagion
        column_names_target = annot.columns
        target_crop_copy = copy.deepcopy(target_crop)
        nr_inds = annot[annot['class'].isin(['tail', 'piglet'])].shape[0]
        annot_target = pd.DataFrame(columns=column_names_target, index=range(0))
        for index, row in annot.iterrows():

            if row["class"]== "tail" or row["class"]== "piglet":
                if target_mask[row["y_center"],row["x_center"],:] == 1:
                    # Adjust the x,y bw, bh values in row
                    #row["x_center"] = int(row["x_center_rel"] * target_crop_dim[1])
                    #row["y_center"] = int(row["y_center_rel"]  * target_crop_dim[0])
                    #row["bw"] = int(row["bw_rel"]  * img_dim[1])
                    #row["bh"] = int(row["bh_rel"]  * img_dim[0])
                    #print(row)
                    delta_x = abs(back_center_x - row["x_center"]) * 3.375
                    delta_y = abs(back_center_y - row["y_center"]) * 3.375
                    #delta_x_rel = delta_x / img_dim[1]
                    #delta_y_rel = delta_y / img_dim[0]
                    # Compute center points
                    x_center_orig = back_center_x - row["x_center"]
                    if x_center_orig <= 0:
                        x_center_crop = int(target_radius2 + delta_x)
                    else:
                        x_center_crop = int(target_radius2 - delta_x)
                    y_center_orig = back_center_y - row["y_center"]
                    if y_center_orig <= 0:
                        y_center_crop = int(target_radius2 + delta_y)
                    else:
                        y_center_crop = int(target_radius2 - delta_y)


                    if row["class_int"] == 3:
                        class_int_val = 0
                    elif row["class_int"] == 2:
                        class_int_val = 1
                    x_center_rel_val = -1
                    y_center_rel_val = -1
                    x_center_val = x_center_crop
                    y_center_val = y_center_crop
                    bw_val = row["bw"] * 3.375
                    bh_val = row["bh"] * 3.375
                    bw_rel_val = -1
                    bh_rel_val = -1
                    class_val = row["class"]
                    new_row  = pd.Series({'class_int': class_int_val,
                                          'x_center_rel': x_center_rel_val,
                                          'y_center_rel': y_center_rel_val,
                                          'x_center': x_center_val,
                                          'y_center': y_center_val,
                                          'bw': bw_val,
                                          'bh': bh_val,
                                          'bw_rel': bw_rel_val,
                                          'bh_rel': bh_rel_val,
                                          'class': class_val}, name='row_name')

                    # Concatenating row-wise
                    annot_target = pd.concat([annot_target, new_row.to_frame().T], axis=0)
                    # Compute box coordinates
                    x_min = int(annot_target.iloc[-1,5 ] - annot_target.iloc[-1,7 ])
                    y_min = int(annot_target.iloc[-1,6 ] - annot_target.iloc[-1,8 ])
                    x_max = int(annot_target.iloc[-1,5 ] + annot_target.iloc[-1,7 ])
                    y_max = int(annot_target.iloc[-1,6 ] + annot_target.iloc[-1,8])
                    ## For manual checking

                    cv2.circle(target_crop_copy, (x_center_crop,y_center_crop), 3, (255,255,255), 1 )
                    cv2.rectangle(target_crop_copy, (int(x_min), int(y_min)),  (int(x_max), int(y_max)), (255,255,255),2)
                    cv2.imwrite(input_folderOut+ "Crop_wAnnotOnly/" + name_only + "Crop_wAnnot.png", target_crop_copy)


        ###########################
        ## Use both the original image and the target crop for the teacher and student model inference
        ## Teacher
        img_teacher_inf = copy.deepcopy(img_orig)
        img_teacher_inf2 = copy.deepcopy(img_orig)
        img_teacher_inf = cv2.cvtColor(img_teacher_inf, cv2.COLOR_BGR2GRAY)
        img_teacher_inf = cv2.cvtColor(img_teacher_inf, cv2.COLOR_GRAY2RGB)
        #print(torch.cuda.current_device())
        result_teacher = model_teacher(img_teacher_inf,verbose=False)
        num_rows_Teacher = len(result_teacher[0].boxes.conf.cpu().numpy())
        # Due to yv8 structure of results -> create empty results pandas df and fill it iteravely
        column_names = ["xmin", "ymin", "xmax", "ymax", "confidence", "class", "name", "x_center", "y_center"]
        # Create an empty DataFrame
        results_teacher_csv = pd.DataFrame(columns=column_names, index=range(num_rows_Teacher))
        # Optionally, you can fill the DataFrame with default values
        # For example, you can fill all columns with zeros
        results_teacher_csv.fillna(-1, inplace=True)
        ## Fill it
        box_info = result_teacher[0].boxes.xyxy.cpu().numpy()
        class_info = result_teacher[0].boxes.cls.cpu().numpy()
        class_info = class_info.astype(dtype=int)
        conf_info = result_teacher[0].boxes.conf.cpu().numpy()
        for res in range(num_rows_Teacher):
            x_min_tmp = box_info[res, 0]
            y_min_tmp = box_info[res, 1]
            x_max_tmp = box_info[res, 2]
            y_max_tmp = box_info[res, 3]
            class_tmp = class_info[res]
            conf_tmp = conf_info[res]
            if class_tmp == 0:
                name_tmp = "head"
            elif class_tmp == 1:
                name_tmp = "back"
            elif class_tmp == 2:
                name_tmp = "piglet"
            elif class_tmp == 3:
                name_tmp = "tail"
            
            teacher_x_center = int((x_min_tmp + x_max_tmp) / 2)
            teacher_y_center = int((y_min_tmp + y_max_tmp) / 2)

            if name_tmp == "tail" or name_tmp== "piglet":
                if target_mask[teacher_y_center,teacher_x_center,:] == 1:

                    delta_xmin = back_center_x - x_min_tmp
                    delta_ymin = back_center_y - y_min_tmp

                    if delta_xmin >= 0:
                        x_min_crop_teacher = int(int(target_radius - abs(delta_xmin) ) * (target_radius2/target_radius))
                    else:
                        x_min_crop_teacher = int(int(target_radius + abs(delta_xmin) ) * (target_radius2/target_radius))

                    if delta_ymin >= 0:
                        y_min_crop_teacher = int(int(target_radius - abs(delta_ymin) ) * (target_radius2/target_radius))
                    else:
                        y_min_crop_teacher = int(int(target_radius + abs(delta_ymin) ) * (target_radius2/target_radius))

                    delta_xmax = back_center_x - x_max_tmp
                    delta_ymax = back_center_y - y_max_tmp
                    if delta_xmax >= 0:
                        x_max_crop_teacher = int(
                            int(target_radius - abs(delta_xmax)) * (target_radius2 / target_radius))
                    else:
                        x_max_crop_teacher = int(
                            int(target_radius + abs(delta_xmax)) * (target_radius2 / target_radius))
                    if delta_ymax >= 0:
                        y_max_crop_teacher = int(
                            int(target_radius - abs(delta_ymax)) * (target_radius2 / target_radius))
                    else:
                        y_max_crop_teacher = int(
                            int(target_radius + abs(delta_ymax)) * (target_radius2 / target_radius))

                    results_teacher_csv.iloc[res, 0] = x_min_crop_teacher #x_min_tmp
                    results_teacher_csv.iloc[res, 1] = y_min_crop_teacher#y_min_tmp
                    results_teacher_csv.iloc[res, 2] = x_max_crop_teacher#x_max_tmp
                    results_teacher_csv.iloc[res, 3] = y_max_crop_teacher#y_max_tmp
                    results_teacher_csv.iloc[res, 4] = conf_tmp
                    results_teacher_csv.iloc[res, 5] = class_tmp
                    results_teacher_csv.iloc[res, 6] = name_tmp
                    results_teacher_csv.iloc[res, 7] = teacher_x_center
                    results_teacher_csv.iloc[res, 8] = teacher_y_center

        ## Student
        target_crop_inf = copy.deepcopy(target_crop)
        target_crop_inf = cv2.resize(target_crop_inf, (1080, 1080))
        target_crop_inf = cv2.copyMakeBorder(target_crop_inf, 0, 0, 0, 2, cv2.BORDER_CONSTANT,value=[255, 255, 255])
        target_crop_inf = cv2.copyMakeBorder(target_crop_inf, 0, 0, 0, 600, cv2.BORDER_CONSTANT, 0)

        
        result_student = model_student(target_crop_inf, verbose=False)
        num_rows_student = len(result_student[0].boxes.conf.cpu().numpy())
        
        # create empty results pandas df and fill it iteravely
        column_names = ["xmin", "ymin", "xmax", "ymax", "confidence", "class", "name"]
        
        # Create an empty DataFrame
        result_student_csv = pd.DataFrame(columns=column_names, index=range(num_rows_student))
        result_student_csv.fillna(-1, inplace=True)
        box_info = result_student[0].boxes.xyxy.cpu().numpy()
        class_info = result_student[0].boxes.cls.cpu().numpy()
        class_info = class_info.astype(dtype=int)
        conf_info = result_student[0].boxes.conf.cpu().numpy()
        for res in range(num_rows_student):
            x_min_tmp = box_info[res, 0]
            y_min_tmp = box_info[res, 1]
            x_max_tmp = box_info[res, 2]
            y_max_tmp = box_info[res, 3]
            class_tmp = class_info[res]
            conf_tmp = conf_info[res]
            if class_tmp == 0:
                name_tmp = "tail"
            elif class_tmp == 1:
                name_tmp = "piglet"
            result_student_csv.iloc[res, 0] = x_min_tmp
            result_student_csv.iloc[res, 1] = y_min_tmp
            result_student_csv.iloc[res, 2] = x_max_tmp
            result_student_csv.iloc[res, 3] = y_max_tmp
            result_student_csv.iloc[res, 4] = conf_tmp
            result_student_csv.iloc[res, 5] = class_tmp
            result_student_csv.iloc[res, 6] = name_tmp

        ## Draw both ground truth and prediction
        target_crop_Empty = copy.deepcopy(target_crop_inf)
        target_crop_combinedStudent = copy.deepcopy(target_crop_inf)
        target_crop_combinedTeacher = copy.deepcopy(img_orig)
        if annot_target.shape[0] == 0:
            with open(file_path_1, 'a', newline='') as file1:
                writer_Detections = csv.writer(file1)
                current_str = [name_only,"annotation","nothing",-1,-1,-1,-1]
                writer_Detections.writerow(current_str)
        for j in range(annot_target.shape[0]):
            color_truth = (255,100,255)
            current_row = annot_target.iloc[j,:]
            current_class = current_row["class"]
            x1 = int(current_row["x_center"] - current_row["bw"])
            y1 = int(current_row["y_center"] - current_row["bh"])
            x2 = int(current_row["x_center"] + current_row["bw"])
            y2 = int(current_row["y_center"] + current_row["bh"])
            # Write the informatione in the overview table
            with open(file_path_1, 'a', newline='') as file1:
                writer_Detections = csv.writer(file1)
                current_str = [name_only,"annotation",current_class,x1,y1,x2,y2]
                writer_Detections.writerow(current_str)
            cv2.rectangle(target_crop_combinedStudent, (x1,y1),(x2,y2), (255,100,255),2)
            cv2.putText(target_crop_combinedStudent, ("Ground truth"), (img_dim[0] + 25, 50), font, 1, color_truth, 1,cv2.LINE_AA)
        
        # student
        for k in range(result_student_csv.shape[0]):
            color_pred = (0,150,250)
            current_row = result_student_csv.iloc[k,:]
            current_class = current_row["name"]
            x1 = int(current_row["xmin"] )
            y1 = int(current_row["ymin"] )
            x2 = int(current_row["xmax"] )
            y2 = int(current_row["ymax"] )
            # Write the informatione in the overview table
            with open(file_path_1, 'a', newline='') as file1:
                writer_Detections = csv.writer(file1)
                current_str = [name_only, "student", current_class, x1, y1, x2, y2]
                writer_Detections.writerow(current_str)
            cv2.rectangle(target_crop_combinedStudent, (x1,y1),(x2,y2), color_pred,2)
            cv2.putText(target_crop_combinedStudent, ("Prediction_Student"),(img_dim[0] + 25, 150), font, 1, color_pred, 1,cv2.LINE_AA)
        
        # teacher
        for l in range(results_teacher_csv.shape[0]):
            color_pred_teacher = (0,255,0)
            current_row = results_teacher_csv.iloc[l,:]
            current_class = current_row["name"]
            if current_row["class"]== 2 or current_row["class"]== 3:
                #x_center, y_center = int((current_row["xmax"]+ current_row["xmin"]) /2), int((current_row["ymax"]+ current_row["ymin"]) /2)
                x_center, y_center = int(current_row["x_center"] ), int(current_row["y_center"])

                if target_mask[y_center,x_center,:] == 1:
                    x1 = int(current_row["xmin"] )
                    y1 = int(current_row["ymin"] )
                    x2 = int(current_row["xmax"] )
                    y2 = int(current_row["ymax"] )
                    # Write the informatione in the overview table
                    with open(file_path_1, 'a', newline='') as file1:
                        writer_Detections = csv.writer(file1)
                        current_str = [name_only, "teacher", current_class, x1, y1, x2, y2]
                        writer_Detections.writerow(current_str)
                    cv2.rectangle(target_crop_combinedTeacher, (x1,y1),(x2,y2), color_pred_teacher,1)

        target_crop_combinedTeacher[inds[0], inds[1]] = 0
        target_crop_combinedTeacher = target_crop_combinedTeacher[target_rect_ymin:target_rect_ymax, target_rect_xmin: target_rect_xmax]
        target_crop_combinedTeacher = cv2.resize(target_crop_combinedTeacher, (img_dim[0], img_dim[0]))
        target_crop_combinedTeacher = cv2.copyMakeBorder(target_crop_combinedTeacher, 0, 0, 0, 2, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        target_crop_combinedTeacher = cv2.copyMakeBorder(target_crop_combinedTeacher, 0, 0, 0, 600, cv2.BORDER_CONSTANT, 0)
        for j in range(annot_target.shape[0]):

            current_row = annot_target.iloc[j,:]
            x1 = int(current_row["x_center"] - current_row["bw"])
            y1 = int(current_row["y_center"] - current_row["bh"])
            x2 = int(current_row["x_center"] + current_row["bw"])
            y2 = int(current_row["y_center"] + current_row["bh"])
            cv2.rectangle(target_crop_combinedTeacher, (x1,y1),(x2,y2), (255,100,255),3)

            cv2.putText(target_crop_combinedTeacher, ("Prediction_Teacher"), (img_dim[0] + 25, 150), font, 1, color_pred_teacher, 1,cv2.LINE_AA)
            cv2.putText(target_crop_combinedTeacher, ("Ground truth"), (img_dim[0] + 25, 50), font, 1, color_truth, 1,cv2.LINE_AA)

    
    evaluation_data = pd.read_csv(file_path_1)
    img_names = evaluation_data.drop_duplicates(subset='img_name')["img_name"]

    ##### [] Compute TPs, FNs, FPs
    count_var = 0
    TP_counter_teacher = 0
    FP_counter_teacher = 0
    FN_counter_teacher = 0
    TP_counter_student = 0
    FP_counter_student = 0
    FN_counter_student = 0

    TP_counter_teacher_tail = 0
    TP_counter_teacher_piglet = 0
    FP_counter_teacher_tail = 0
    FP_counter_teacher_piglet = 0
    FN_counter_teacher_tail = 0
    FN_counter_teacher_piglet = 0
    TP_counter_student_tail = 0
    TP_counter_student_piglet = 0
    FP_counter_student_tail = 0
    FP_counter_student_piglet = 0
    FN_counter_student_tail = 0
    FN_counter_student_piglet = 0

    for eval_nr in range(0,len(img_names)):
        eval_img = list(img_names)[eval_nr]
        empty_target_img = cv2.imread(input_folderOut+ "EmptyTargetFrames/" + eval_img + ".png", 1)
        count_var += 1

        # Filter csv for this image
        subdata = evaluation_data[evaluation_data["img_name"] == eval_img]
        subdata_gt = subdata[subdata["mode"].str.contains("annotation", na=False)]
        subdata_teacher = subdata[subdata["mode"].str.contains("teacher", na=False)]
        subdata_student = subdata[subdata["mode"].str.contains("student", na=False)]

        if  subdata_gt["class"].eq("nothing").any():
            no_gt = True
        else:
            no_gt = False

        if len(subdata_teacher) == 0:
            no_teacher = True
        else:
            no_teacher = False

        if len(subdata_student) == 0:
            no_student = True
        else:
            no_student = False

        if no_gt:
            subdata_gt.drop(index=subdata_gt.index, inplace=True)

        # Copy new dataframes to map objects
        subdata_gt_assigned = copy.deepcopy(subdata_gt)
        subdata_teacher_assigned = copy.deepcopy(subdata_teacher)
        subdata_student_assigned = copy.deepcopy(subdata_student)

        subdata_gt_assigned["Assigned_teacher"] = "unassigned"
        subdata_gt_assigned["Assigned_teacher_id"] = -1
        subdata_gt_assigned["Assigned_student"] = "unassigned"
        subdata_gt_assigned["Assigned_student_id"] = -1
        subdata_teacher_assigned["Assigned"] = "unassigned"
        subdata_student_assigned["Assigned"] = "unassigned"
        subdata_teacher_assigned["IOU"] = -1
        subdata_student_assigned["IOU"] = -1



        # Resetting the index
        subdata_gt_assigned.reset_index(drop=True, inplace=True)
        subdata_teacher_assigned.reset_index(drop=True, inplace=True)
        subdata_student_assigned.reset_index(drop=True, inplace=True)

        # Create masks to quantify IoU
        empty_frame = np.zeros(shape=(empty_target_img.shape[0], empty_target_img.shape[1], 1))
        target_copy = copy.deepcopy(empty_target_img)

        #### TP und FN
        for gt_object_i in range(len(subdata_gt)):
            empty_frame_copy = copy.deepcopy(empty_frame)
            current_gt_class = subdata_gt.iloc[gt_object_i, :]["class"]
            x1_tmp_gt =  subdata_gt.iloc[gt_object_i, :]["x_min"]
            y1_tmp_gt = subdata_gt.iloc[gt_object_i, :]["y_min"]
            x2_tmp_gt = subdata_gt.iloc[gt_object_i, :]["x_max"]
            y2_tmp_gt = subdata_gt.iloc[gt_object_i, :]["y_max"]
            gt_mask = cv2.rectangle(empty_frame_copy, (x1_tmp_gt, y1_tmp_gt),(x2_tmp_gt,y2_tmp_gt), (1), -1)
            cv2.rectangle(target_copy, (x1_tmp_gt, y1_tmp_gt),(x2_tmp_gt,y2_tmp_gt), (255,255,255), 2)


            
            teacher_iou_mark = [-1,-1] 
            for teacher_object_i in range(len(subdata_teacher_assigned)):
                empty_frame_copy_t = copy.deepcopy(empty_frame)
                current_teacher_class = subdata_teacher_assigned.iloc[teacher_object_i, :]["class"]
                if subdata_teacher_assigned.iloc[teacher_object_i, :]["Assigned"] == "assigned":
                    already_assigned_t = True
                else:
                    already_assigned_t = False


                if current_gt_class == current_teacher_class and already_assigned_t == False:
                    x1_tmp_t = subdata_teacher_assigned.iloc[teacher_object_i, :]["x_min"]
                    y1_tmp_t = subdata_teacher_assigned.iloc[teacher_object_i, :]["y_min"]
                    x2_tmp_t = subdata_teacher_assigned.iloc[teacher_object_i, :]["x_max"]
                    y2_tmp_t = subdata_teacher_assigned.iloc[teacher_object_i, :]["y_max"]
                    empty_frame_copy_t2 = copy.deepcopy(empty_frame)
                    teacher_mask = cv2.rectangle(empty_frame_copy_t2, (x1_tmp_t, y1_tmp_t), (x2_tmp_t, y2_tmp_t), (1), -1)
                    cv2.rectangle(target_copy, (x1_tmp_t, y1_tmp_t), (x2_tmp_t, y2_tmp_t), (0, 0, 255), 2)
                    # Compare with target_mask
                    gt_mask_copy_t = copy.deepcopy(gt_mask)
                    comb_mask_gt_teacher = gt_mask_copy_t + teacher_mask
                    # calculate IoU
                    pixels_1_t = len(np.where(comb_mask_gt_teacher == 1)[0])
                    pixels_2_t = len(np.where(comb_mask_gt_teacher == 2)[0])
                    current_IoU_t = pixels_2_t / (pixels_1_t + pixels_2_t)

                    if current_IoU_t >= IoU_thresh_1 and current_IoU_t >= teacher_iou_mark[0]:
                        teacher_iou_mark[0] = current_IoU_t
                        teacher_iou_mark[1] = teacher_object_i

            
            if teacher_iou_mark[1] != -1:
                subdata_gt_assigned.loc[gt_object_i,"Assigned_teacher"] = "assigned"
                subdata_gt_assigned.loc[gt_object_i, "Assigned_teacher_id"] = teacher_iou_mark[1]
                subdata_teacher_assigned.loc[teacher_iou_mark[1], "Assigned"] = "assigned"
                subdata_teacher_assigned.loc[teacher_iou_mark[1], "IOU"] = teacher_iou_mark[0]


            
            student_iou_mark_1 = [-1,-1]  
            for student_object_i in range(len(subdata_student_assigned)):
                empty_frame_copy_s = copy.deepcopy(empty_frame)
                current_student_class = subdata_student_assigned.iloc[student_object_i, :]["class"]
                if subdata_student_assigned.iloc[student_object_i, :]["Assigned"] == "assigned":
                    already_assigned_s = True
                else:
                    already_assigned_s = False

                if current_gt_class == current_student_class and already_assigned_s == False:
                    x1_tmp_s = subdata_student_assigned.iloc[student_object_i, :]["x_min"]
                    y1_tmp_s = subdata_student_assigned.iloc[student_object_i, :]["y_min"]
                    x2_tmp_s = subdata_student_assigned.iloc[student_object_i, :]["x_max"]
                    y2_tmp_s = subdata_student_assigned.iloc[student_object_i, :]["y_max"]
                    empty_frame_copy_s2 = copy.deepcopy(empty_frame)
                    student_mask = cv2.rectangle(empty_frame_copy_s2, (x1_tmp_s, y1_tmp_s), (x2_tmp_s, y2_tmp_s), (1), -1)
                    cv2.rectangle(target_copy, (x1_tmp_s, y1_tmp_s), (x2_tmp_s, y2_tmp_s), (10, 255, 30), 2)
                    # Compare with target_mask
                    gt_mask_copy_s = copy.deepcopy(gt_mask)
                    comb_mask_gt_student = gt_mask_copy_s + student_mask
                    # calculate IoU
                    pixels_1_s = len(np.where(comb_mask_gt_student == 1)[0])
                    pixels_2_s = len(np.where(comb_mask_gt_student == 2)[0])
                    current_IoU_s = pixels_2_s / (pixels_1_s + pixels_2_s)

                    if current_IoU_s >= IoU_thresh_1 and current_IoU_s >= student_iou_mark_1[0]:
                        student_iou_mark_1[0] = current_IoU_s
                        student_iou_mark_1[1] = student_object_i

            if student_iou_mark_1[1] != -1:
                subdata_gt_assigned.loc[gt_object_i, "Assigned_student"] = "assigned"
                subdata_gt_assigned.loc[gt_object_i, "Assigned_student_id"] = student_iou_mark_1[1]
                subdata_student_assigned.loc[student_iou_mark_1[1], "Assigned"] = "assigned"
                subdata_student_assigned.loc[student_iou_mark_1[1], "IOU"] = student_iou_mark_1[0]

        #### FP
        # Teacher
        for teacher_object_j in range(len(subdata_teacher_assigned )):
            empty_frame_copy_FP_t = copy.deepcopy(empty_frame)
            current_teacher_class = subdata_teacher_assigned.iloc[teacher_object_j, :]["class"]
            current_t_pred_matched = False
            x1_tmp_t =  subdata_teacher_assigned.iloc[teacher_object_j, :]["x_min"]
            y1_tmp_t = subdata_teacher_assigned.iloc[teacher_object_j, :]["y_min"]
            x2_tmp_t = subdata_teacher_assigned.iloc[teacher_object_j, :]["x_max"]
            y2_tmp_t = subdata_teacher_assigned.iloc[teacher_object_j, :]["y_max"]
            teacher_mask = cv2.rectangle(empty_frame_copy_FP_t, (x1_tmp_t, y1_tmp_t),(x2_tmp_t,y2_tmp_t), (1), -1)

            
            gt_iou_mark = [-1,-1] 
            for gt_object_j in range(len(subdata_gt_assigned)):
                empty_frame_copy_FP_gt = copy.deepcopy(empty_frame)
                current_gt_class = subdata_gt_assigned.iloc[gt_object_j, :]["class"]

                if current_teacher_class == current_gt_class:
                    current_t_pred_matched = True 
                    x1_tmp = subdata_gt_assigned.iloc[gt_object_j, :]["x_min"]
                    y1_tmp = subdata_gt_assigned.iloc[gt_object_j, :]["y_min"]
                    x2_tmp = subdata_gt_assigned.iloc[gt_object_j, :]["x_max"]
                    y2_tmp = subdata_gt_assigned.iloc[gt_object_j, :]["y_max"]
                    empty_frame_copy_t3 = copy.deepcopy(empty_frame)
                    gt_mask = cv2.rectangle(empty_frame_copy_t3, (x1_tmp, y1_tmp), (x2_tmp, y2_tmp), (1), -1)
                    # Compare with target_mask
                    teacher_mask_copy = copy.deepcopy(teacher_mask)
                    comb_mask_gt_teacher = gt_mask + teacher_mask_copy
                    # calculate IoU
                    pixels_1 = len(np.where(comb_mask_gt_teacher == 1)[0])
                    pixels_2 = len(np.where(comb_mask_gt_teacher == 2)[0])
                    current_IoU_t2 = pixels_2 / (pixels_1 + pixels_2)

                    if current_IoU_t2 >= IoU_thresh_1 and current_IoU_t2 >= gt_iou_mark[0]:
                        gt_iou_mark[0] = current_IoU_t2
                        gt_iou_mark[1] = gt_object_j

                if current_t_pred_matched == False:
                    cv2.rectangle(target_copy, (x1_tmp_t, y1_tmp_t), (x2_tmp_t, y2_tmp_t), (0, 0, 255), 2)

            
            if gt_iou_mark[1] != -1 and subdata_gt_assigned.loc[gt_iou_mark[1], "Assigned_teacher_id"] == -1:
                subdata_teacher_assigned.loc[teacher_object_j, "Assigned"] = "assigned"
                subdata_teacher_assigned.loc[teacher_object_j, "IOU"] = gt_iou_mark[0]

        #Student
        for student_object_j in range(len(subdata_student_assigned)):
            empty_frame_copy_FP_s = copy.deepcopy(empty_frame)
            current_student_class = subdata_student_assigned.iloc[student_object_j, :]["class"]
            current_s_pred_matched = False
            x1_tmp_s = subdata_student_assigned.iloc[student_object_j, :]["x_min"]
            y1_tmp_s = subdata_student_assigned.iloc[student_object_j, :]["y_min"]
            x2_tmp_s = subdata_student_assigned.iloc[student_object_j, :]["x_max"]
            y2_tmp_s = subdata_student_assigned.iloc[student_object_j, :]["y_max"]
            
            student_mask = cv2.rectangle(empty_frame_copy_FP_s, (int(x1_tmp_s), int(y1_tmp_s)), (int(x2_tmp_s), int(y2_tmp_s)), (1), -1)

            
            gt_iou_mark = [-1,-1]  
            for gt_object_j in range(len(subdata_gt_assigned)):
                empty_frame_copy_FP_gt = copy.deepcopy(empty_frame)
                current_gt_class = subdata_gt_assigned.iloc[gt_object_j, :]["class"]

                if current_student_class == current_gt_class:
                    current_s_pred_matched = True  
                    x1_tmp = subdata_gt_assigned.iloc[gt_object_j, :]["x_min"]
                    y1_tmp = subdata_gt_assigned.iloc[gt_object_j, :]["y_min"]
                    x2_tmp = subdata_gt_assigned.iloc[gt_object_j, :]["x_max"]
                    y2_tmp = subdata_gt_assigned.iloc[gt_object_j, :]["y_max"]
                    empty_frame_copy_s3 = copy.deepcopy(empty_frame)
                    gt_mask = cv2.rectangle(empty_frame_copy_s3, (int(x1_tmp), int(y1_tmp)), (int(x2_tmp), int(y2_tmp)), (1), -1)
                    
                    # Compare with target_mask
                    comb_mask_gt_student = gt_mask + student_mask
                    
                    # calculate IoU
                    pixels_1 = len(np.where(comb_mask_gt_student == 1)[0])
                    pixels_2 = len(np.where(comb_mask_gt_student == 2)[0])
                    current_IoU_s2 = pixels_2 / (pixels_1 + pixels_2)

                    if current_IoU_s2 >= IoU_thresh_1 and current_IoU_s2 >= gt_iou_mark[0]:
                        gt_iou_mark[0] = current_IoU_s2
                        gt_iou_mark[1] = gt_object_j

                if current_s_pred_matched == False:
                    cv2.rectangle(target_copy, (int(x1_tmp_s), int(y1_tmp_s)), (int(x2_tmp_s), int(y2_tmp_s)), (0, 0, 255), 2)

            
            if gt_iou_mark[1] != -1 and subdata_gt_assigned.loc[gt_iou_mark[1], "Assigned_student_id"] == -1 :
                subdata_student_assigned.loc[student_object_j, "Assigned"] = "assigned"
                subdata_student_assigned.loc[student_object_j, "IOU"] = gt_iou_mark[0]


        # Calculate metrics for current image
        #TPs
        TP_img_teacher =  len(subdata_gt_assigned[subdata_gt_assigned["Assigned_teacher"] == 'assigned'])
        TP_img_student = len(subdata_gt_assigned[subdata_gt_assigned["Assigned_student"] == 'assigned'])

        TP_img_teacher_piglet = len(subdata_gt_assigned[(subdata_gt_assigned["Assigned_teacher"] == 'assigned') & (subdata_gt_assigned["class"] == 'piglet')])
        TP_img_teacher_tail = len(subdata_gt_assigned[(subdata_gt_assigned["Assigned_teacher"] == 'assigned') & (subdata_gt_assigned["class"] == 'tail')])
        TP_img_student_piglet = len(subdata_gt_assigned[(subdata_gt_assigned["Assigned_student"] == 'assigned') & (subdata_gt_assigned["class"] == 'piglet')])
        TP_img_student_tail = len(subdata_gt_assigned[(subdata_gt_assigned["Assigned_student"] == 'assigned') & (subdata_gt_assigned["class"] == 'tail')])


        # FNs
        FN_img_teacher = len(subdata_gt_assigned[subdata_gt_assigned["Assigned_teacher"] == 'unassigned'])
        FN_img_student = len(subdata_gt_assigned[subdata_gt_assigned["Assigned_student"] == 'unassigned'])

        FN_img_teacher_piglet = len(subdata_gt_assigned[(subdata_gt_assigned["Assigned_teacher"] == 'unassigned') & (
                    subdata_gt_assigned["class"] == 'piglet')])
        FN_img_teacher_tail = len(subdata_gt_assigned[(subdata_gt_assigned["Assigned_teacher"] == 'unassigned') & (
                    subdata_gt_assigned["class"] == 'tail')])
        FN_img_student_piglet = len(subdata_gt_assigned[(subdata_gt_assigned["Assigned_student"] == 'unassigned') & (
                    subdata_gt_assigned["class"] == 'piglet')])
        FN_img_student_tail = len(subdata_gt_assigned[(subdata_gt_assigned["Assigned_student"] == 'unassigned') & (
                    subdata_gt_assigned["class"] == 'tail')])

        # FPs
        FP_img_teacher = len(subdata_teacher_assigned[subdata_teacher_assigned["Assigned"] == 'unassigned'])
        FP_img_student = len(subdata_student_assigned[subdata_student_assigned["Assigned"] == 'unassigned'])

        FP_img_teacher_piglet = len(subdata_teacher_assigned[(subdata_teacher_assigned["Assigned"] == 'unassigned') & (subdata_teacher_assigned["class"] == 'piglet')])
        FP_img_teacher_tail = len(subdata_teacher_assigned[(subdata_teacher_assigned["Assigned"] == 'unassigned') & (
                    subdata_teacher_assigned["class"] == 'tail')])
        FP_img_student_piglet = len(subdata_student_assigned[(subdata_student_assigned["Assigned"] == 'unassigned') & (
                    subdata_student_assigned["class"] == 'piglet')])
        FP_img_student_tail = len(subdata_student_assigned[(subdata_student_assigned["Assigned"] == 'unassigned') & (
                    subdata_student_assigned["class"] == 'tail')])


        TP_counter_teacher = TP_counter_teacher + TP_img_teacher
        FP_counter_teacher = FP_counter_teacher + FP_img_teacher
        FN_counter_teacher = FN_counter_teacher + FN_img_teacher

        TP_counter_student = TP_counter_student + TP_img_student
        FP_counter_student = FP_counter_student + FP_img_student
        FN_counter_student = FN_counter_student + FN_img_student

        TP_counter_teacher_tail = TP_counter_teacher_tail + TP_img_teacher_tail
        TP_counter_teacher_piglet = TP_counter_teacher_piglet + TP_img_teacher_piglet
        FP_counter_teacher_tail = FP_counter_teacher_tail + FP_img_teacher_tail
        FP_counter_teacher_piglet = FP_counter_teacher_piglet +FP_img_teacher_piglet
        FN_counter_teacher_tail = FN_counter_teacher_tail + FN_img_teacher_tail
        FN_counter_teacher_piglet = FN_counter_teacher_piglet + FN_img_teacher_piglet
        TP_counter_student_tail = TP_counter_student_tail + TP_img_student_tail
        TP_counter_student_piglet = TP_counter_student_piglet + TP_img_student_piglet
        FP_counter_student_tail = FP_counter_student_tail + FP_img_student_tail
        FP_counter_student_piglet = FP_counter_student_piglet + FP_img_student_piglet
        FN_counter_student_tail = FN_counter_student_tail + FN_img_student_tail
        FN_counter_student_piglet = FN_counter_student_piglet + FN_img_student_piglet

        gt_color = (255,255,255)
        teacher_color = (0,0,255)
        student_color = (0,255,0)

        for gt_object_i in range(len(subdata_gt_assigned)):
            current_gt_class = subdata_gt_assigned.iloc[gt_object_i, :]["class"]

            x1_tmp = subdata_gt_assigned.iloc[gt_object_i, :]["x_min"]
            y1_tmp = subdata_gt_assigned.iloc[gt_object_i, :]["y_min"]
            x2_tmp = subdata_gt_assigned.iloc[gt_object_i, :]["x_max"]
            y2_tmp = subdata_gt_assigned.iloc[gt_object_i, :]["y_max"]
            cv2.rectangle(empty_target_img, (x1_tmp, y1_tmp), (x2_tmp, y2_tmp), gt_color, 2)
            cv2.putText(empty_target_img, current_gt_class, (int(x1_tmp), int(y1_tmp+22)),font, 1, gt_color,2)

        for gt_object_i in range(len(subdata_teacher_assigned)):
            current_gt_class = subdata_teacher_assigned.iloc[gt_object_i, :]["class"]
            assignment_stat = subdata_teacher_assigned.iloc[gt_object_i, :]["Assigned"]
            current_iou = subdata_teacher_assigned.iloc[gt_object_i, :]["IOU"]
            x1_tmp = subdata_teacher_assigned.iloc[gt_object_i, :]["x_min"]
            y1_tmp = subdata_teacher_assigned.iloc[gt_object_i, :]["y_min"]
            x2_tmp = subdata_teacher_assigned.iloc[gt_object_i, :]["x_max"]
            y2_tmp = subdata_teacher_assigned.iloc[gt_object_i, :]["y_max"]
            cv2.rectangle(empty_target_img, (int(x1_tmp), int(y1_tmp)), (int(x2_tmp), int(y2_tmp)), teacher_color, 2)
            if assignment_stat == "assigned":
                cv2.putText(empty_target_img, current_gt_class +"_" + str(round(current_iou,3)), (int(x1_tmp), int( y2_tmp - 22)),font, 1, teacher_color,2)
            else:
                cv2.putText(empty_target_img, current_gt_class + "_(unassigned)" +"_" + str(round(current_iou,3)), (int(x1_tmp), int(y2_tmp - 22)), font,1, teacher_color, 2)

        for gt_object_i in range(len(subdata_student_assigned)):
            current_gt_class = subdata_student_assigned.iloc[gt_object_i, :]["class"]
            assignment_stat = subdata_student_assigned.iloc[gt_object_i, :]["Assigned"]
            current_iou = subdata_student_assigned.iloc[gt_object_i, :]["IOU"]
            x1_tmp = subdata_student_assigned.iloc[gt_object_i, :]["x_min"]
            y1_tmp = subdata_student_assigned.iloc[gt_object_i, :]["y_min"]
            x2_tmp = subdata_student_assigned.iloc[gt_object_i, :]["x_max"]
            y2_tmp = subdata_student_assigned.iloc[gt_object_i, :]["y_max"]
            cv2.rectangle(empty_target_img, (x1_tmp, y1_tmp), (x2_tmp, y2_tmp), student_color, 2)
            if assignment_stat == "assigned":
                cv2.putText(empty_target_img, current_gt_class +"_" + str(round(current_iou,3)), (x1_tmp, y2_tmp + 22),font, 1, student_color,2)
            else:
                cv2.putText(empty_target_img, current_gt_class + "_(unassigned)_" + str(round(current_iou,3)), (x1_tmp, y2_tmp + 22), font, 1, student_color, 2)



        # Make a boarder
        empty_target_img = cv2.copyMakeBorder(empty_target_img, 150,0,0,0,cv2.BORDER_CONSTANT, (0,0,0))
        # Add the color legends
        cv2.circle(empty_target_img, (50,50), 10, teacher_color, -1)
        cv2.circle(empty_target_img, (50, 100), 10, student_color, -1)
        cv2.putText(empty_target_img, "T", (75, 50), font, 0.7, teacher_color, 2)
        cv2.putText(empty_target_img, "S", (75, 100), font, 0.7, student_color,2)

        # Add number of TP, FP FN
        cv2.putText(empty_target_img, f"TP: {str(TP_img_teacher)}; FN: {str(FN_img_teacher)}; FP: {str(FP_img_teacher)}; TPT {str(TP_img_teacher_tail)}, FPT {str(FP_img_teacher_tail)}, FNT {str(FN_img_teacher_tail)}, TPP {str(TP_img_teacher_piglet)}, FPP {str(FP_img_teacher_piglet)}, FNP {str(FN_img_teacher_piglet)}", (150, 50), font, 0.7, teacher_color, 2)
        cv2.putText(empty_target_img, f"TP: {str(TP_img_student)}; FN: {str(FN_img_student)}; FP: {str(FP_img_student)}; TPT {str(TP_img_student_tail)}, FPT {str(FP_img_student_tail)}, FNT {str(FN_img_student_tail)}, TPP {str(TP_img_student_piglet)}, FPP {str(FP_img_student_piglet)}, FNP {str(FN_img_student_piglet)}", (150, 100), font, 0.7,student_color, 2)

        #plt.imshow(empty_target_img)

        # Visualize the results from the teacher and student model and combine the information
        img_teacher = cv2.imread(input_evalSet + "/images/" + eval_img + ".png",1)
        img_teacher = cv2.cvtColor(img_teacher, cv2.COLOR_BGR2GRAY)
        img_teacher = cv2.cvtColor(img_teacher, cv2.COLOR_GRAY2RGB)
        result_teacher = model_teacher(img_teacher, verbose=False)
        for r in result_teacher:
            im_array_t = r.plot()
        img_student = cv2.imread(input_folderOut + "EmptyTargetFrames/" + eval_img + ".png", 1)
        img_student = cv2.resize(img_student, (1080, 1080))
        img_student = cv2.copyMakeBorder(img_student, 0, 0, 0, 2, cv2.BORDER_CONSTANT, value=[255, 255, 255])
        img_student = cv2.copyMakeBorder(img_student, 0, 0, 0, 600, cv2.BORDER_CONSTANT, 0)
        result_student = model_student(img_student, verbose=False)
        for s in result_student:
            im_array_s = s.plot()
        im_array_t = cv2.resize(im_array_t, (540,540))
        im_array_s = cv2.resize(im_array_s, (540,540))
        img_comb = np.concatenate((im_array_t, im_array_s), axis=1)
        img_comb = np.concatenate((img_comb, empty_target_img), axis=0)
        cv2.imwrite(input_folderOut + "ResultFrames/" + eval_img + ".png", img_comb)





    # Calculate metric for whole dataset
    recall_s = round(TP_counter_student / (TP_counter_student + FN_counter_student),3)
    recall_t = round(TP_counter_teacher / (TP_counter_teacher + FN_counter_teacher),3)
    recall_t_t = round(TP_counter_teacher_tail / (TP_counter_teacher_tail + FN_counter_teacher_tail),3)
    recall_t_p = round(TP_counter_teacher_piglet / (TP_counter_teacher_piglet + FN_counter_teacher_piglet), 3)
    recall_s_t = round(TP_counter_student_tail / (TP_counter_student_tail + FN_counter_student_tail), 3)
    recall_s_p = round(TP_counter_student_piglet / (TP_counter_student_piglet + FN_counter_student_piglet), 3)

    precision_s = round(TP_counter_student / (TP_counter_student + FP_counter_student),3)
    precision_t = round(TP_counter_teacher / (TP_counter_teacher + FP_counter_teacher),3)
    precision_t_t = round(TP_counter_teacher_tail / (TP_counter_teacher_tail + FP_counter_teacher_tail), 3)
    precision_t_p = round(TP_counter_teacher_piglet / (TP_counter_teacher_piglet + FP_counter_teacher_piglet), 3)
    precision_s_t = round(TP_counter_student_tail / (TP_counter_student_tail + FP_counter_student_tail), 3)
    precision_s_p = round(TP_counter_student_piglet / (TP_counter_student_piglet + FP_counter_student_piglet), 3)

    f1_s = round(TP_counter_student / (TP_counter_student + ((FP_counter_student + FN_counter_student)/2)) ,3)
    f1_t = round(TP_counter_teacher / (TP_counter_teacher + ((FP_counter_teacher + FN_counter_teacher) / 2)), 3)
    f1_t_t = round(TP_counter_teacher_tail / (TP_counter_teacher_tail + ((FP_counter_teacher_tail + FN_counter_teacher_tail)/2)), 3)
    f1_t_p = round(TP_counter_teacher_piglet / (TP_counter_teacher_piglet + ((FP_counter_teacher_piglet + FN_counter_teacher_piglet)/2)), 3)
    f1_s_t = round(TP_counter_student_tail / (TP_counter_student_tail + ((FP_counter_student_tail + FN_counter_student_tail)/2)), 3)
    f1_s_p = round(TP_counter_student_piglet / (TP_counter_student_piglet + ((FP_counter_student_piglet + FN_counter_student_piglet)/2)), 3)


    # Create output csv
    columns = ('Evaluation_name','Modeltype','Recall','Precision','F1_score','TP_total', 'FP_total', 'FN_total', 'TPT', 'FPT', 'FNT', 'TPP', 'FPP', 'FNP', 'Recall_T', 'Precision_T', 'F1_T', 'Recall_P', 'Precision_P', 'F1_P')


    data = [
        [folder_name,"teacher" ,recall_t, precision_t, f1_t, TP_counter_teacher, FP_counter_teacher,FN_counter_teacher,TP_counter_teacher_tail,FP_counter_teacher_tail,FN_counter_teacher_tail,TP_counter_teacher_piglet,FP_counter_teacher_piglet,FN_counter_teacher_piglet,recall_t_t, precision_t_t,f1_t_t,recall_t_p,precision_t_p, f1_t_p   ],
        [folder_name, "student", recall_s, precision_s, f1_s, TP_counter_student, FP_counter_student,FN_counter_student,TP_counter_student_tail,FP_counter_student_tail,FN_counter_student_tail,TP_counter_student_piglet,FP_counter_student_piglet,FN_counter_student_piglet,recall_s_t, precision_s_t,f1_s_t,recall_s_p,precision_s_p,f1_s_p ]]

    # Create a DataFrame
    metrics_df = pd.DataFrame(data, columns=columns)
    # Specify the file name
    file_name = input_folderOut +'metrics.csv'
    # Save the DataFrame to a CSV file without including the index column
    metrics_df.to_csv(file_name, index=False)


    print("Evaluation finished")
    print(f"\nFinal scores: \n Teacher: Recall: {recall_t}, Precision: {precision_t}, F1-score: {f1_t}, TP: {TP_counter_teacher}, FP{FP_counter_teacher}, FN: {FN_counter_teacher} \n Student: Recall: {recall_s}, Precision: {precision_s}, F1-score: {f1_s}, TP: {TP_counter_student}, FP{FP_counter_student}, FN: {FN_counter_student}")

def main():
    parser = argparse.ArgumentParser(description='Compare the performance of the student and teacher model for piglet detection.')
    parser.add_argument('path_teacher', type=str, help='Input path of teacher weights.')
    parser.add_argument('path_student',type=str, help='Input path of student weights.')
    parser.add_argument('path_evaluation', type=str, help='Folder path of evaluation images and txt files in separate folders.')
    parser.add_argument('output_folder', type=str, help='Path of output folder')

    args = parser.parse_args()

    evaluate(args.path_teacher, args.path_student, args.path_evaluation, args.output_folder)

if __name__ == '__main__':
    main()
