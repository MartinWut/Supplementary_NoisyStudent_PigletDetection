import copy

import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import pandas as pd
import timeit
import random
import shutil
from ultralytics import YOLO



# Create parser
parser = argparse.ArgumentParser()
subparsers = parser.add_subparsers(help='Commands', dest='command')
parser_convert = subparsers.add_parser('video_analysis', help='Analyse and render a video file')
parser_convert.add_argument('--video_path', help='The full path of the input video.', required=True)
parser_convert.add_argument('--model_path', help='The full path of the model pt-file.', required=True)
parser_convert.add_argument('--model_path_target', help='The full path of the target model pt-file.', required=True)
parser_convert.add_argument('--output_name',
                            help='OPTIONAL, The name and path of the output video. \n If not specified output is saved as "output_file.avi" in the working directory.',
                            required=False)
parser_convert.add_argument('--radius',
                            help='Integer. Size of target radius',
                            required=True, type = int)
parser_convert.add_argument('--fps',
                            help='Integer. FPS rate for extracion',
                            required=True, type = int)


args = parser.parse_args()
video_path = args.video_path
crop_case = args.crop_case

reverse_creation = args.reverse
model = YOLO(args.model_path)
model_target  = YOLO(args.model_path_target)
if args.output_name is not None:
    output_name = args.output_name
else:
    output_name = "output_video.avi"





# target_area_size
target_area_size = args.radius
target_fps = args.fps

conf_piglet = 0.1
conf_tail = 0.1
conf_pt_tail = 0.1
conf_pt_piglet = 0.1


video_cap = cv2.VideoCapture(video_path)
fps_video = video_cap.get(cv2.CAP_PROP_FPS)
fps_video = round(fps_video)
total_frames = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # float `width`
height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))  # float `height`



video_writer = cv2.VideoWriter((output_name), cv2.VideoWriter_fourcc(*'XVID'), fps_video, (width, height),
                                   isColor=True)

print(
    "\nThe video was successfully loaded from: %s \n FPS: %s \n number of frames: %s \n Dimension: (%s,%s) \n output path: %s" % (
    video_path, str(fps_video), str(total_frames), str(width), str(height), (output_name)))

color_vec = [(255, 0, 0), (0, 0, 255), (0, 255, 0), (125, 0, 255)]  # [(255,0,0),(0,255,0), (0,0,255)]
class_labels = ["head", "back", "piglet", "tail"]
class_labels_target = ["tail", "piglet"]
font = cv2.FONT_HERSHEY_DUPLEX
box_thickness = 2

piglet_list = []
count = 0
count_pig_list = 0
tob = None
init_head_pos = None
init_tail_pos = None
signal_object_head = None
signal_object_tail = None
signal_object_tail_target = None


# create a new folder for the video
newPath = output_folder + "cropped_target_area/" + video_name[::-1]
isExist = os.path.exists(newPath)
if not isExist:
    # Create a new directory because it does not exist
    os.makedirs(newPath)
    os.makedirs(newPath + "/tail_only/images")
    os.makedirs(newPath + "/tail_only/images_control")
    os.makedirs(newPath + "/tail_only/annots")
    os.makedirs(newPath + "/piglet_only/images")
    os.makedirs(newPath + "/piglet_only/images_control")
    os.makedirs(newPath + "/piglet_only/annots")
    os.makedirs(newPath + "/piglet_tail/images")
    os.makedirs(newPath + "/piglet_tail/images_control")
    os.makedirs(newPath + "/piglet_tail/annots")
    os.makedirs(newPath + "/multi_piglets/images")
    os.makedirs(newPath + "/multi_piglets/images_control")
    os.makedirs(newPath + "/multi_piglets/annots")
    os.makedirs(newPath + "/multi_piglets_tail/images")
    os.makedirs(newPath + "/multi_piglets_tail/images_control")
    os.makedirs(newPath + "/multi_piglets_tail/annots")
    os.makedirs(newPath + "/target_only/images")
    os.makedirs(newPath + "/target_only/annots")

    print("The new directory is created at %s!" % (newPath))

count_pt = 0


while (video_cap.isOpened()):
    ret, frame = video_cap.read()
    frame_dim = frame.shape

    while (ret == True):
        # take only every n-th frame
        if count % target_fps != 0:
            count += 1
            ret, frame = video_cap.read()

            if (count >= (total_frames - 1)) & ((ret != True)):
                print("release video")
                # if args.split == "default":
                video_writer.release()
                video_cap.release()
                ret = False
            continue

        else:
            if count % 100 == 0:
                print(count)
            
            no_head = False
            no_back = False

            # make inference
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

            frame_copy_2 = copy.deepcopy(frame)

            results_yv8 = model(frame,  verbose=False)
            
            num_rows = len(results_yv8[0].boxes.conf.cpu().numpy())

            if num_rows == 0:
                print("Skipping step")
                if count >= (total_frames - 1):
                    print("release video")
                    # if args.split == "default":
                    video_writer.release()
                    video_cap.release()
                    ret = False
                
                status_code = 0
                status_text = "inactive - no detection results (%s)" % (str(status_code))
            else:

                # set loop indicators
                head_processed = False
                back_processed = False
                head_back_open = True

                column_names = ["xmin", "ymin", "xmax", "ymax", "confidence", "class", "name"]
                
                results_csv = pd.DataFrame(columns=column_names, index=range(num_rows))
               
                results_csv.fillna(-1, inplace=True)
               
                box_info = results_yv8[0].boxes.xyxy.cpu().numpy()
                class_info = results_yv8[0].boxes.cls.cpu().numpy()
                class_info = class_info.astype(dtype=int)
                conf_info = results_yv8[0].boxes.conf.cpu().numpy()

                for res in range(num_rows):
                    x_min_tmp = box_info[res,0]
                    y_min_tmp = box_info[res,1]
                    x_max_tmp = box_info[res,2]
                    y_max_tmp = box_info[res,3]
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

                    results_csv.iloc[res,0] = x_min_tmp
                    results_csv.iloc[res, 1] = y_min_tmp
                    results_csv.iloc[res, 2] = x_max_tmp
                    results_csv.iloc[res, 3] = y_max_tmp
                    results_csv.iloc[res, 4] = conf_tmp
                    results_csv.iloc[res, 5] = class_tmp
                    results_csv.iloc[res, 6] = name_tmp

               
                previous_pigList = piglet_list
                class_array = np.array(results_csv["class"])
                conf_array = np.array(results_csv["confidence"])
                nr_piglet = len(np.where((class_array == 2) & (conf_array > 0.8))[0])
                
                nr_piglet_toLose = np.where((class_array == 2) & (conf_array <= 0.8))[0]
                
                for i in nr_piglet_toLose:
                    results_csv_t = results_csv.T
                    popped_row = results_csv_t.pop(i)
                    results_csv = results_csv_t.T

                results_csv = results_csv.sort_values(by="class", ascending=True)
                class_array = np.array(results_csv["class"])
                conf_array = np.array(results_csv["confidence"])
                nr_heads = len(np.where(class_array == 0)[0])
                nr_backs = len(np.where(class_array == 1)[0])
                nr_tails = len(np.where(class_array == 3)[0])

                skipped_detections = []

                if nr_heads > 1:

                    
                    row_inds = list(np.where(class_array == 0)[0])
                    max_ind_head = row_inds[np.where(
                        np.array(results_csv.iloc[row_inds, 4] == np.max(np.array(results_csv.iloc[row_inds, 4]))))[0][
                        0]]
                    skipped_rows = row_inds[:max_ind_head] + row_inds[max_ind_head + 1:]
                    for i in skipped_rows:
                        skipped_detections.append(i)
                    
                    head_center_x = int(
                        (int(results_csv.iloc[max_ind_head, 0]) + int(results_csv.iloc[max_ind_head, 2])) / 2)
                    head_center_y = int(
                        (int(results_csv.iloc[max_ind_head, 3]) + int(results_csv.iloc[max_ind_head, 1])) / 2)
                    csv_row_head = results_csv.iloc[max_ind_head]
                    signal_object_head = 1
                elif nr_heads == 1:
                    
                    row_inds = list(np.where(class_array == 0)[0])[0]
                    
                    head_center_x = int((int(results_csv.iloc[row_inds, 0]) + int(results_csv.iloc[row_inds, 2])) / 2)
                    head_center_y = int((int(results_csv.iloc[row_inds, 3]) + int(results_csv.iloc[row_inds, 1])) / 2)
                    csv_row_head = results_csv.iloc[row_inds]
                    signal_object_head = 1
                else:
                    
                    count += 1
                    no_head = True
                    

                
                if nr_backs > 1:

                    row_inds = list(np.where(class_array == 1)[0])
                    max_ind_back = row_inds[np.where(
                        np.array(results_csv.iloc[row_inds, 4] == np.max(np.array(results_csv.iloc[row_inds, 4]))))[0][
                        0]]
                    skipped_rows = row_inds[:max_ind_back] + row_inds[max_ind_back + 1:]
                    for i in skipped_rows:
                        skipped_detections.append(i)
                    
                    back_center_x = int(
                        (int(results_csv.iloc[max_ind_back, 0]) + int(results_csv.iloc[max_ind_back, 2])) / 2)
                    back_center_y = int(
                        (int(results_csv.iloc[max_ind_back, 3]) + int(results_csv.iloc[max_ind_back, 1])) / 2)
                    csv_row_back = results_csv.iloc[max_ind_back]
                    signal_object_back = 1
                elif nr_backs == 1:
                    row_inds = list(np.where(class_array == 1)[0])[0]
                    
                    back_center_x = int((int(results_csv.iloc[row_inds, 0]) + int(results_csv.iloc[row_inds, 2])) / 2)
                    back_center_y = int((int(results_csv.iloc[row_inds, 3]) + int(results_csv.iloc[row_inds, 1])) / 2)
                    csv_row_back = results_csv.iloc[row_inds]
                    signal_object_back = 1
                else:  
                    count += 1
                    no_back = True
                    
                
                if nr_tails > 1:

                    row_inds = list(np.where(class_array == 3)[0])
                    max_ind_tail = row_inds[np.where(
                        np.array(results_csv.iloc[row_inds, 4] == np.max(np.array(results_csv.iloc[row_inds, 4]))))[0][
                        0]]
                    skipped_rows = row_inds[:max_ind_tail] + row_inds[max_ind_tail + 1:]
                    for i in skipped_rows:
                        skipped_detections.append(i)
                    # compute tail center
                    tail_center_x = int(
                        (int(results_csv.iloc[max_ind_tail, 0]) + int(results_csv.iloc[max_ind_tail, 2])) / 2)
                    tail_center_y = int(
                        (int(results_csv.iloc[max_ind_tail, 3]) + int(results_csv.iloc[max_ind_tail, 1])) / 2)
                    csv_row_tail = results_csv.iloc[max_ind_tail]
                    signal_object_tail = 1
                elif nr_tails == 1:
                    row_inds = list(np.where(class_array == 3)[0])[0]
                    
                    tail_center_x = int((int(results_csv.iloc[row_inds, 0]) + int(results_csv.iloc[row_inds, 2])) / 2)
                    tail_center_y = int((int(results_csv.iloc[row_inds, 3]) + int(results_csv.iloc[row_inds, 1])) / 2)
                    csv_row_tail = results_csv.iloc[row_inds]
                    signal_object_tail = 1

               
                for row in range(len(results_csv)):

                    if no_head or no_back:  # render frame only with current classes
                        x_min, y_min, x_max, y_max = results_csv.iloc[row, 0], results_csv.iloc[row, 1], \
                        results_csv.iloc[row, 2], results_csv.iloc[row, 3]
                        current_class = results_csv.iloc[row, 5]  # class 0=head; 1=tail; 2=piglet
                        current_conf = round(results_csv.iloc[row, 4], 2)
                        
                        frame = cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)),
                                              color_vec[current_class], 3)
                        
                        current_label_info = (class_labels[current_class] + " " + str(current_conf))
                        text_size = cv2.getTextSize(current_label_info, cv2.FONT_HERSHEY_SIMPLEX, 0.8, cv2.LINE_AA)
                        frame = cv2.rectangle(frame, (int(x_min), int(y_min - 20)),
                                              (int(x_min) + text_size[0][0], int(y_min)),
                                              color_vec[current_class], -1)
                        
                        frame = cv2.putText(frame, (class_labels[current_class] + " " + str(current_conf)),
                                            (int(x_min + 5), int(y_min)), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                            (255, 255, 255), 2,
                                            cv2.LINE_AA)
                        if no_head == True & no_back == False:
                            status_code = 1
                            status_text = "no head detection (%s)" % (str(status_code))
                        elif no_head == False & no_back == True:
                            status_code = 2
                            status_text = "no back detection (%s)" % (str(status_code))
                        else:
                            status_code = 3
                            status_text = "no head and back detection (%s)" % (str(status_code))
                    else:
                        # set the status parameters
                        status_code = 0
                        status_text = "active (%s)" % (str(status_code))
                        if row in skipped_detections:
                            
                            continue

                        x_min, y_min, x_max, y_max = results_csv.iloc[row, 0], results_csv.iloc[row, 1], \
                        results_csv.iloc[row, 2], results_csv.iloc[row, 3]
                        current_class = results_csv.iloc[row, 5]  # class 0=head; 1 = back;  2=piglet; 3=tail
                        current_conf = round(results_csv.iloc[row, 4], 2)

                       
                        frame = cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)),
                                              color_vec[current_class], box_thickness)
                        
                        current_label_info = (class_labels[current_class] + " " + str(current_conf))
                        text_size = cv2.getTextSize(current_label_info, font, 0.8, cv2.LINE_AA)
                        frame = cv2.rectangle(frame, (int(x_min), int(y_min - 20)),
                                              (int(x_min) + text_size[0][0], int(y_min)),
                                              color_vec[current_class], -1)
                        
                        frame = cv2.putText(frame, current_label_info, (int(x_min + 5), int(y_min)), font,
                                            0.8, (255, 255, 255), 2, cv2.LINE_AA)

                        if current_class == 0:
                            head_processed = True
                        if current_class == 1:
                            back_processed = True

                        if head_processed & back_processed & head_back_open:

                            head_back_open = False
                            
                            frame = cv2.line(frame, (head_center_x, head_center_y), (back_center_x, back_center_y),
                                             color=(255, 0, 0),
                                             thickness=box_thickness)
                            
                            target_radius = target_area_size
                            frame = cv2.circle(frame, (back_center_x, back_center_y), target_radius, (0, 0, 255), 2)

                            piglet_detec = None
                            tail_detec = None
                            pt_detec = None




                            empty_frame = np.zeros(shape=(frame.shape[0], frame.shape[1], 1))

                            mask = cv2.circle(empty_frame, (back_center_x, back_center_y), target_radius, (1), -1)
                            inds = np.where(mask == 0)
                            frame_copy_2[inds[0], inds[1], 0] = 0
                            frame_copy_2[inds[0], inds[1], 1] = 0
                            frame_copy_2[inds[0], inds[1], 2] = 0

                            target_rect_ymin, target_rect_xmin, target_rect_ymax, target_rect_xmax = int(
                                back_center_y - target_radius), int(back_center_x - target_radius), int(
                                back_center_y + target_radius), int(back_center_x + target_radius)
                            target_focus_crop = frame_copy_2[target_rect_ymin:target_rect_ymax,
                                                target_rect_xmin: target_rect_xmax]
                            target_center_x, target_center_y = target_radius, target_radius
                            target_focus_crop_dim = target_focus_crop.shape

                            target_focus_crop_noAnnos = copy.deepcopy(target_focus_crop)

                            if (target_focus_crop.shape[0] == 2 * target_radius) and (target_focus_crop.shape[1] == 2 * target_radius):
                                results_yv8_target = model_target(target_focus_crop,  verbose=False)
                                num_rows_target = len(results_yv8_target[0].boxes.conf.cpu().numpy())

                                
                                column_names = ["xmin", "ymin", "xmax", "ymax", "confidence", "class", "name"]
                               
                                results_target_csv = pd.DataFrame(columns=column_names, index=range(num_rows_target))
                               
                                results_target_csv.fillna(-1, inplace=True)
                             
                                box_info = results_yv8_target[0].boxes.xyxy.cpu().numpy()
                                class_info = results_yv8_target[0].boxes.cls.cpu().numpy()
                                class_info = class_info.astype(dtype=int)
                                conf_info = results_yv8_target[0].boxes.conf.cpu().numpy()

                                for res in range(num_rows_target):
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

                                    results_target_csv.iloc[res, 0] = x_min_tmp
                                    results_target_csv.iloc[res, 1] = y_min_tmp
                                    results_target_csv.iloc[res, 2] = x_max_tmp
                                    results_target_csv.iloc[res, 3] = y_max_tmp
                                    results_target_csv.iloc[res, 4] = conf_tmp
                                    results_target_csv.iloc[res, 5] = class_tmp
                                    results_target_csv.iloc[res, 6] = name_tmp
                                

                                skipped_detections_target = []

                                if len(results_target_csv) != 0:
                                   
                                    results_target_csv = results_target_csv.sort_values(by="class", ascending=True)
                                    class_array_target = np.array(results_target_csv["class"])
                                    conf_array_target = np.array(results_target_csv["confidence"])
                                    nr_tails_target = len(np.where(class_array_target == 0)[0])
                                    nr_piglets_target = len(np.where(class_array_target == 1)[0])
                                   
                                    multi_piglets_count = 0
                                    multi_tails = False

                                    if (nr_tails_target <= 1):
                                        signal_object_tail_target = 1

                                    if nr_tails_target > 1:
                                        print("\nMultiple tails in target area")
                                        multi_tails = True
                                        row_inds_target = list(np.where(class_array_target == 0)[0])
                                        
                                        max_ind_tail_target = row_inds_target[np.where(np.array(
                                            results_target_csv.iloc[row_inds_target, 4] == np.max(np.array(results_target_csv.iloc[row_inds_target, 4]))))[0][0]]
                                        skipped_rows = row_inds_target[:max_ind_tail_target] + row_inds_target[  max_ind_tail_target + 1:]
                                        for i in skipped_rows:
                                            skipped_detections_target.append(i)
                                        
                                        tail_center_x_target = int((int(results_target_csv.iloc[max_ind_tail_target, 0]) + int( results_target_csv.iloc[max_ind_tail_target, 2])) / 2)
                                        tail_center_y_target = int((int( results_target_csv.iloc[max_ind_tail_target, 3]) + int( results_target_csv.iloc[max_ind_tail_target, 1])) / 2)
                                        csv_row_tail_target = results_target_csv.iloc[max_ind_tail_target]
                                        signal_object_tail_target = 1


                                    
                                    current_space = 10
                                    if nr_piglets_target >= 1:
                                        piglet_detec = True
                                    if ( signal_object_tail_target == 1) and nr_tails_target >= 1:
                                        tail_detec = True

                                    if piglet_detec and not tail_detec:
                                        prefix = "piglet_"
                                        conf_piglet_tmp = 0
                                        str_list_piglet = []
                                        str_list_piglet_total = []
                                        for row in range(len(results_target_csv)):
                                            if row in skipped_detections_target:
                                                continue
                                            else:
                                                x_min_target, y_min_target, x_max_target, y_max_target = results_target_csv.iloc[row, 0], results_target_csv.iloc[row, 1],  results_target_csv.iloc[row, 2], results_target_csv.iloc[ row, 3]
                                                current_class_target = results_target_csv.iloc[ row, 5]  # class 0=tail; 1=piglet
                                                current_conf_target = round(results_target_csv.iloc[row, 4], 2)

                                                if (current_class_target == 1) and (current_conf_target >= conf_piglet):
                                                   
                                                    multi_piglets_count += 1
                                                    conf_piglet_tmp = current_conf_target
                                                    piglet_xmin, piglet_ymin, piglet_xmax, piglet_ymax = x_min_target, y_min_target, x_max_target, y_max_target
                                                    piglet_box_width_rel, piglet_box_height_rel = int(x_max_target - x_min_target) / int(target_radius * 2), int(y_max_target - y_min_target) / int(target_radius * 2)
                                                    piglet_x_center_rel, piglet_y_center_rel = (int((piglet_xmin + piglet_xmax) / 2)) / int(2 * target_radius),(int((piglet_ymin + piglet_ymax) / 2)) / int(2 * target_radius)
                                                    target_focus_crop = cv2.rectangle(target_focus_crop, (int(piglet_xmin), int(piglet_ymin)), (int(piglet_xmax),int(piglet_ymax)),color_vec[current_class_target],box_thickness)
                                                    
                                                    current_label_info_target = (class_labels_target[ current_class_target] + " " + str( current_conf_target))
                                                    text_size_target = cv2.getTextSize( current_label_info_target, font, 0.8, cv2.LINE_AA)
                                                    str_list_piglet = [str(piglet_x_center_rel), str(piglet_y_center_rel),str(piglet_box_width_rel), str(piglet_box_height_rel)]
                                                    str_list_piglet_total.append(str_list_piglet)
                                                   
                                                    target_focus_crop = cv2.putText(target_focus_crop, current_label_info_target, ( int(piglet_xmin + 5),int(piglet_ymin)), font, 0.4, (255, 255, 255), 1,cv2.LINE_AA)
                                                    target_focus_crop = cv2.circle(target_focus_crop, (10, 0 + current_space), 10,color_vec[current_class_target], -1 )
                                                    current_space += 25

                                                   
                                        if (conf_piglet_tmp >= conf_pt_piglet):

                                            if multi_piglets_count > 1:
                                                prefix = "MultiPiglets_" + str(multi_piglets_count) + "_"
                                                print("\n Multiple piglets detected at frame: " + str(count))
                                                cv2.imwrite( newPath + "/multi_piglets/images_control/" + prefix + video_name[  ::-1] + str( count) + ".png", target_focus_crop)
                                                cv2.imwrite(newPath + "/multi_piglets/images/" + prefix + video_name[    ::-1] + str( count) + ".png", target_focus_crop_noAnnos)
                                                q = open(newPath + "/multi_piglets/annots/" + prefix + video_name[ ::-1] + str( count) + '.txt', "a")
                                                
                                                print(len(str_list_piglet_total))
                                                for entry in range(len(str_list_piglet_total)):
                                                    q.writelines([str(1) + " ", ' '.join(str_list_piglet_total[entry])])
                                                    q.writelines(["\n"])
                                                q.close()
                                            elif multi_piglets_count == 1:
                                                prefix = "PT_detec_"
                                                cv2.imwrite(newPath + "/piglet_only/images/" + prefix + video_name[::-1] + str(count) + ".png", target_focus_crop_noAnnos)
                                                cv2.imwrite(newPath + "/piglet_only/images_control/" + prefix + video_name[::-1] + str(count) + ".png", target_focus_crop)
                                                q = open(newPath + "/piglet_tail/annots/" + prefix + video_name[::-1] + str(count) + '.txt', "a")
                                                q.writelines([str(1) + " ", ' '.join(str_list_piglet)])
                                                q.writelines(["\n"])
                                                q.close()
                                            else:
                                                continue


                                    elif tail_detec and not piglet_detec:
                                        prefix = "tail_"
                                        filename = prefix + video_name[::-1] + "_" + str(count)

                                        for row in range(len(results_target_csv)):
                                            if row in skipped_detections_target:
                                                continue
                                            else:
                                                x_min_target, y_min_target, x_max_target, y_max_target = results_target_csv.iloc[row, 0], results_target_csv.iloc[row, 1],  results_target_csv.iloc[row, 2], results_target_csv.iloc[ row, 3]
                                                current_class_target = results_target_csv.iloc[ row, 5]  # class 0=tail; 1=piglet
                                                current_conf_target = round(results_target_csv.iloc[row, 4], 2)

                                                if (current_class_target == 0) and (current_conf_target >= conf_tail):
                                                    tail_xmin, tail_ymin, tail_xmax, tail_ymax = x_min_target, y_min_target, x_max_target, y_max_target
                                                    tail_box_width_rel, tail_box_height_rel = int(tail_xmax - tail_xmin) / int(target_radius * 2), int(tail_ymax - tail_ymin) / int(target_radius * 2)
                                                    tail_x_center_rel, tail_y_center_rel = (int((tail_xmin + tail_xmax) / 2)) / int(2 * target_radius),(int((tail_ymin + tail_ymax) / 2)) / int(2 * target_radius)

                                                    target_focus_crop = cv2.rectangle(target_focus_crop, (int(tail_xmin), int(tail_ymin)), (int(tail_xmax),   int(tail_ymax)), color_vec[current_class_target],box_thickness)
                                                   
                                                    current_label_info_target = (class_labels_target[current_class_target] + " " + str(  current_conf_target))
                                                    text_size_target = cv2.getTextSize(    current_label_info_target, font, 0.8, cv2.LINE_AA)
                                                    
                                                    target_focus_crop = cv2.putText(target_focus_crop,   current_label_info_target, (   int(tail_xmin + 5),   int(tail_ymin)), font,  0.4, (255, 255, 255), 1, cv2.LINE_AA)

                                                    cv2.imwrite( newPath + "/tail_only/images_control/" + filename + ".png",target_focus_crop)
                                                    cv2.imwrite(newPath + "/tail_only/images/" + filename + ".png", target_focus_crop_noAnnos)
                                                    l = open(newPath + "/tail_only/annots/" + filename+ '.txt', "a")
                                                    str_list = [str(tail_x_center_rel), str(tail_y_center_rel), str(tail_box_width_rel), str(tail_box_height_rel)]
                                                    l.writelines([str(0) + " ", ' '.join(str_list)])
                                                    l.writelines(["\n"])


                                    elif piglet_detec and tail_detec:
                                        conf_piglet_tmp = 0
                                        conf_tail_tmp = 0
                                        str_list_piglet = []
                                        str_list_piglet_total = []
                                        
                                        for row in range(len(results_target_csv)):
                                            if row in skipped_detections_target:
                                                continue
                                            else:
                                                x_min_target, y_min_target, x_max_target, y_max_target = results_target_csv.iloc[row, 0], results_target_csv.iloc[row, 1],  results_target_csv.iloc[row, 2], results_target_csv.iloc[ row, 3]
                                                current_class_target = results_target_csv.iloc[ row, 5]  # class 0=tail; 1=piglet
                                                current_conf_target = round(results_target_csv.iloc[row, 4], 2)

                                                
                                                
                                                if (current_class_target == 0) and (current_conf_target >= conf_pt_tail):
                                                    conf_tail_tmp = current_conf_target
                                                    tail_xmin, tail_ymin, tail_xmax, tail_ymax = x_min_target, y_min_target, x_max_target, y_max_target
                                                    tail_box_width_rel, tail_box_height_rel = int(tail_xmax - tail_xmin) / int(target_radius * 2), int(tail_ymax - tail_ymin) / int(target_radius * 2)
                                                    tail_x_center_rel, tail_y_center_rel = (int(( tail_xmin + tail_xmax) / 2)) / int(2 * target_radius), ( int((tail_ymin + tail_ymax) / 2)) / int(2 * target_radius)
                                                    str_list_tail = [str(tail_x_center_rel), str(tail_y_center_rel), str(tail_box_width_rel), str(tail_box_height_rel)]

                                                    target_focus_crop = cv2.rectangle(target_focus_crop, (  int(tail_xmin), int(tail_ymin)), (int(tail_xmax), int(tail_ymax)), color_vec[ current_class_target],box_thickness)
                                                    
                                                    current_label_info_target = (class_labels_target[   current_class_target] + " " + str(current_conf_target))
                                                    text_size_target = cv2.getTextSize(current_label_info_target,  font, 0.8, cv2.LINE_AA)
                                                    current_space += 25

                                                if (current_class_target == 1) and (current_conf_target >= conf_pt_piglet):
                                                    multi_piglets_count += 1
                                                    conf_piglet_tmp = current_conf_target
                                                    piglet_xmin, piglet_ymin, piglet_xmax, piglet_ymax = x_min_target, y_min_target, x_max_target, y_max_target
                                                    piglet_box_width_rel, piglet_box_height_rel = int(piglet_xmax - piglet_xmin) / int(target_radius * 2), int( piglet_ymax - piglet_ymin) / int(target_radius * 2)
                                                    piglet_x_center_rel, piglet_y_center_rel = (int((piglet_xmin + piglet_xmax) / 2)) / int(2 * target_radius), (int(( piglet_ymin + piglet_ymax) / 2)) / int(2 * target_radius)
                                                    str_list_piglet = [str(piglet_x_center_rel), str(piglet_y_center_rel),str(piglet_box_width_rel), str(piglet_box_height_rel)]
                                                    str_list_piglet_total.append(str_list_piglet)
                                                    target_focus_crop = cv2.rectangle(target_focus_crop,(int(piglet_xmin), int(piglet_ymin)),   (int(piglet_xmax), int(piglet_ymax)),  color_vec[current_class_target], box_thickness)
                                                    
                                                    current_label_info_target = (   class_labels_target[current_class_target] + " " + str( current_conf_target))
                                                    text_size_target = cv2.getTextSize(current_label_info_target, font, 0.8,    cv2.LINE_AA)
                                                    current_space += 25

                                        if (conf_piglet_tmp >= conf_pt_piglet) and (conf_tail_tmp >= conf_pt_tail):

                                            if multi_piglets_count > 1:
                                                prefix = "MultiPiglets_" + str(multi_piglets_count) + "_"
                                                print("\n Multiple piglets detected at frame: " + str(count))
                                                cv2.imwrite( newPath + "/multi_piglets_tail/images_control/" + prefix + video_name[  ::-1] + str( count) + ".png", target_focus_crop)
                                                cv2.imwrite(newPath + "/multi_piglets_tail/images/" + prefix + video_name[    ::-1] + str( count) + ".png", target_focus_crop_noAnnos)
                                                q = open(newPath + "/multi_piglets_tail/annots/" + prefix + video_name[ ::-1] + str( count) + '.txt', "a")
                                                q.writelines([str(0) + " ", ' '.join(str_list_tail)])
                                                q.writelines(["\n"])
                                                
                                                print(len(str_list_piglet_total))
                                                for entry in range(len(str_list_piglet_total)):
                                                    q.writelines([str(1) + " ", ' '.join(str_list_piglet_total[entry])])
                                                    q.writelines(["\n"])
                                                q.close()
                                            elif multi_piglets_count == 1:
                                                prefix = "PT_detec_"
                                                cv2.imwrite(newPath + "/piglet_tail/images/" + prefix + video_name[::-1] + str(count) + ".png", target_focus_crop_noAnnos)
                                                cv2.imwrite(newPath + "/piglet_tail/images_control/" + prefix + video_name[::-1] + str(count) + ".png", target_focus_crop)
                                                q = open(newPath + "/piglet_tail/annots/" + prefix + video_name[::-1] + str(count) + '.txt', "a")
                                                q.writelines([str(0) + " ", ' '.join(str_list_tail)])
                                                q.writelines(["\n"])
                                                q.writelines([str(1) + " ", ' '.join(str_list_piglet)])
                                                q.writelines(["\n"])
                                                q.close()
                                            else:
                                                continue


                                    else:
                                        prefix = "targetArea_"
                                        cv2.imwrite( newPath + "/target_only/images/" + prefix + video_name[::-1] + str(count) + ".png", target_focus_crop)



            video_writer.write(frame)
            ret, frame = video_cap.read()

            if count >= (total_frames - 1):
                print("release video")
                # if args.split == "default":
                video_writer.release()
                video_cap.release()
                ret = False

            count += 1
