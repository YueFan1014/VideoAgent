import cv2
from ultralytics import YOLO, FastSAM, SAM, RTDETR, NAS
from time import time
import pickle
import os
from collections import defaultdict
import clip
import random as rd
from PIL import Image
import torch
import numpy as np
rd.seed(0)
import torchvision.transforms as T
from PIL import Image
from time import time
import math
import sys
from io import StringIO


id2category = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    4: 'airplane',
    5: 'bus',
    6: 'train',
    7: 'truck',
    8: 'boat',
    9: 'traffic light',
    10: 'fire hydrant',
    11: 'stop sign',
    12: 'parking meter',
    13: 'bench',
    14: 'bird',
    15: 'cat',
    16: 'dog',
    17: 'horse',
    18: 'sheep',
    19: 'cow',
    20: 'elephant',
    21: 'bear',
    22: 'zebra',
    23: 'giraffe',
    24: 'backpack',
    25: 'umbrella',
    26: 'handbag',
    27: 'tie',
    28: 'suitcase',
    29: 'frisbee',
    30: 'skis',
    31: 'snowboard',
    32: 'sports ball',
    33: 'kite',
    34: 'baseball bat',
    35: 'baseball glove',
    36: 'skateboard',
    37: 'surfboard',
    38: 'tennis racket',
    39: 'bottle',
    40: 'wine glass',
    41: 'cup',
    42: 'fork',
    43: 'knife',
    44: 'spoon',
    45: 'bowl',
    46: 'banana',
    47: 'apple',
    48: 'sandwich',
    49: 'orange',
    50: 'broccoli',
    51: 'carrot',
    52: 'hot dog',
    53: 'pizza',
    54: 'donut',
    55: 'cake',
    56: 'chair',
    57: 'couch',
    58: 'potted plant',
    59: 'bed',
    60: 'dining table',
    61: 'toilet',
    62: 'tv',
    63: 'laptop',
    64: 'mouse',
    65: 'remote',
    66: 'keyboard',
    67: 'cell phone',
    68: 'microwave',
    69: 'oven',
    70: 'toaster',
    71: 'sink',
    72: 'refrigerator',
    73: 'book',
    74: 'clock',
    75: 'vase',
    76: 'scissors',
    77: 'teddy bear',
    78: 'hair drier',
    79: 'toothbrush'
}


class Tracking:
    def __init__(self, video_path_list, base_dir='preprocess',tracking_fps=30, sample_num=10, show=False):
        self.video_path_list = video_path_list
        self.base_dir = base_dir
        self.tracking_fps = tracking_fps
        self.sample_num = sample_num
        self.show = show


    def object_tracking(self):
        print("Start object tracking...")
        for video_path in self.video_path_list:
            base_name = os.path.basename(video_path).replace(".mp4", "")
            video_dir = os.path.join(self.base_dir, base_name)
            if not os.path.exists(video_dir):
                os.makedirs(video_dir)

            start_time = time()
            model = RTDETR('tool_models/tracking/rtdetr-l.pt')

            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            video_fps = round(cap.get(cv2.CAP_PROP_FPS))
            frame_interval = max(1, math.ceil(video_fps / self.tracking_fps))
            frame2trackid = defaultdict(dict)
            trackid2frame = defaultdict(list)
            trackid2category_cnt = defaultdict(list)
            # Loop through the video frames
            frame_idx = -1
            while cap.isOpened():
                success, frame = cap.read()
                frame_idx += 1
                if not success:
                    break
                if frame_idx % frame_interval == 0: 
                    results = model.track(frame, persist=True, tracker="bytetrack.yaml", verbose=False)
                    boxes = results[0].boxes
                    if boxes.id is None:
                        continue
                    cls = boxes.cls.numpy()
                    id = boxes.id.numpy()
                    xywh = boxes.xywh.numpy() #notice: x, y are the center coordinates
                    num_boxes = id.shape[0]
                    for i in range(num_boxes):
                        track_id = int(id[i])
                        category = id2category[cls[i]]
                        box = list(xywh[i])
                        frame2trackid[frame_idx][track_id] = [category, box]
                        trackid2frame[track_id].append(frame_idx)
                        trackid2category_cnt[track_id].append(category)
                    if self.show:
                        annotated_frame = results[0].plot()
                        cv2.imshow("RTDETR Tracking", annotated_frame)
                        cv2.waitKey(10)
            cap.release()
            cv2.destroyAllWindows()
            trackid2category = dict()
            for track_id in trackid2category_cnt:
                category_cnt = trackid2category_cnt[track_id]
                most_common_category = max(set(category_cnt), key=category_cnt.count)
                trackid2category[track_id] = most_common_category

            tracking_file = os.path.join(video_dir, 'tracking.pkl')
            with open(tracking_file, 'wb') as f:
                pickle.dump([frame2trackid, trackid2frame, trackid2category], f)
            end_time = time()
            print(f'tracking time for video {base_name}: {round(end_time-start_time, 3)} seconds')


    def create_CLIP_and_DINOv2_embedding(self):
        """generate the CLIP and DINOv2 embedding for each tracking ID"""
        clip_model, clip_transform = clip.load("tool_models/CLIP/ViT-B-32.pt", device="cuda")
        dinov2_model = torch.hub.load('tool_models/facebookresearch_dinov2_main', 'dinov2_vitg14', source="local").cuda()
        dinov2_transform = T.Compose([
            T.Resize(256, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ])

        for video_path in self.video_path_list:
            base_name = os.path.basename(video_path).replace(".mp4", "")
            video_dir = os.path.join(self.base_dir, base_name)
            start_time = time()
            cap = cv2.VideoCapture(video_path)
            tracking_file = os.path.join(video_dir, 'tracking.pkl')
            with open(tracking_file, 'rb') as f:
                temp = pickle.load(f)
            frame2trackid, trackid2frame = temp[0], temp[1]
            frame2select = defaultdict(list)
            track_id2clip_emb = defaultdict(list)
            track_id2dinov2_emb = defaultdict(list)
            for track_id in trackid2frame:
                frame_ids = trackid2frame[track_id]
                selected_frame_ids = rd.sample(frame_ids, min(len(frame_ids), self.sample_num))
                for frame in selected_frame_ids:
                    frame2select[frame].append(track_id)

            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_idx = -1
            while cap.isOpened():
                success, frame = cap.read()
                frame_idx += 1
                if not success:
                    break
                if frame_idx not in frame2select:
                    continue
                for track_id in frame2select[frame_idx]:
                    bbox = frame2trackid[frame_idx][track_id][1]
                    x, y, w, h = bbox
                    left_top_x = int(x-w/2)
                    left_top_y = int(y-h/2)
                    right_bottom_x = int(x+w/2)
                    right_bottom_y = int(y+h/2)
                    cropped_region = frame[left_top_y:right_bottom_y, left_top_x:right_bottom_x] 
                    resized_cropped_region = cv2.resize(cropped_region, (224, 224))
                    img = cv2.cvtColor(resized_cropped_region, cv2.COLOR_BGR2RGB)
                    img = Image.fromarray(img)
                    clip_input = clip_transform(img).unsqueeze(0).cuda()
                    dinov2_input = dinov2_transform(img).unsqueeze(0).cuda()
                    with torch.no_grad():
                        clip_feature = clip_model.encode_image(clip_input)
                        dinov2_feature = dinov2_model(dinov2_input)
                    track_id2clip_emb[track_id].append(clip_feature)
                    track_id2dinov2_emb[track_id].append(dinov2_feature)
            cap.release()
            track_id2avg_clip_emb = dict()
            track_id2avg_dinov2_emb = dict()
            for track_id in track_id2clip_emb:
                clip_emb = torch.cat(track_id2clip_emb[track_id], dim=0)
                average_clip_emb = torch.mean(clip_emb, dim=0).cpu()
                track_id2avg_clip_emb[track_id] = average_clip_emb
            for track_id in track_id2dinov2_emb:
                dinov2_emb = torch.cat(track_id2dinov2_emb[track_id], dim=0)
                average_dinov2_emb = torch.mean(dinov2_emb, dim=0).cpu()
                track_id2avg_dinov2_emb[track_id] = average_dinov2_emb

            save_file = os.path.join(video_dir, 'tid2clip.pkl')
            with open(save_file, 'wb') as f:
                pickle.dump(track_id2avg_clip_emb, f)
            save_file = os.path.join(video_dir, 'tid2dinov2.pkl')
            with open(save_file, 'wb') as f:
                pickle.dump(track_id2avg_dinov2_emb, f)
            end_time = time()
            print(f"object embedding time for video {base_name}: {round(end_time-start_time, 3)} seconds")


    def run(self):
        self.object_tracking()
        self.create_CLIP_and_DINOv2_embedding()
