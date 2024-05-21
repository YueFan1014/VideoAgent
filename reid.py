import cv2
from time import time
import json
import pickle
import os
from collections import defaultdict
import clip
import random as rd
from PIL import Image
import torch
import numpy as np
import imageio
rd.seed(0)


def hash_color(obj_id):
    np.random.seed(obj_id)
    color = np.random.randint(0, 256, 3)
    new_color = tuple(int(i) for i in color)
    return new_color
    

class ReID:
    def __init__(self, video_path_list, base_dir='preprocess'):
        self.video_path_list = video_path_list
        self.base_dir = base_dir
        self.trackid2clip_emb = None
        self.trackid2dinov2_emb = None
        self.trackid2frame = None
        self.trackid2category = None
        self.uid2tids = None
        self.tid2uid = None


    def hard_constraint(self, obj1, obj2):
        # if self.trackid2category[obj1] != self.trackid2category[obj2]: # if two tracked objects have different categories, they cannot be the same object 
        #     return False
        frame1 = set(self.trackid2frame[obj1])
        frame2 = set(self.trackid2frame[obj2])
        if len(frame1.intersection(frame2)) > 0: # if two tracked objects co-exist, they cannot be the same object
            return False
        return True
    

    def clip_similarity_score(self, obj1, obj2, x0=0.925, slope=20):
        clip_emb1 = self.trackid2clip_emb[obj1]
        clip_emb2 = self.trackid2clip_emb[obj2]
        cosine_score = np.dot(clip_emb1, clip_emb2) / (np.linalg.norm(clip_emb1) * np.linalg.norm(clip_emb2))
        clip_score = 1 / (1 + np.exp(-slope * (cosine_score - x0)))
        return clip_score


    def dinov2_similarity_score(self, obj1, obj2, x0=0.5, slope=4.1):
        dinov2_emb1 = self.trackid2dinov2_emb[obj1]
        dinov2_emb2 = self.trackid2dinov2_emb[obj2]
        cosine_score = np.dot(dinov2_emb1, dinov2_emb2) / (np.linalg.norm(dinov2_emb1) * np.linalg.norm(dinov2_emb2))
        #dinov2_score = 1 / (1 + np.exp(-slope * (cosine_score - x0)))
        dinov2_score = cosine_score
        return dinov2_score
    
    
    def compute_score(self, obj1, obj2):
        if not self.hard_constraint(obj1, obj2):
            return 0
        clip_score = self.clip_similarity_score(obj1, obj2)
        dinov2_score = self.dinov2_similarity_score(obj1, obj2)
        return 0.15*clip_score+ 0.85*dinov2_score


    def check_group(self, tid, uid):
        """tid should has score > 0.5 for all uid objects, and at least one score > 0.62"""
        sgn = False
        for t in self.uid2tids[uid]:
            if self.compute_score(tid, t) < 0.5:
                return False
            if self.compute_score(tid, t) >= 0.62:
                sgn = True
        return sgn
    

    def reid_for_all_videos(self):
        for video_path in self.video_path_list:
            base_name = os.path.basename(video_path).replace(".mp4", "")
            video_dir = os.path.join(self.base_dir, base_name)
            with open(os.path.join(video_dir, 'tid2clip.pkl'), 'rb') as f:
                self.trackid2clip_emb = pickle.load(f)
            with open(os.path.join(video_dir, 'tid2dinov2.pkl'), 'rb') as f:
                self.trackid2dinov2_emb = pickle.load(f)
            with open(os.path.join(video_dir, 'tracking.pkl'), 'rb') as f:
                content = pickle.load(f)
            self.frame2trackid, self.trackid2frame, self.trackid2category = content[0], content[1], content[2]
            self.uid2tids = defaultdict(list)
            self.tid2uid = dict()

            for frame in self.frame2trackid:
                cur_track_ids = self.frame2trackid[frame]
                for tid in cur_track_ids:
                    if tid in self.tid2uid:
                        continue
                    sgn = False
                    for uid in self.uid2tids:
                        if self.check_group(tid, uid):
                            self.uid2tids[uid].append(tid)
                            self.tid2uid[tid] = uid
                            sgn = True
                            break
                    if sgn == False:
                        uid = len(self.uid2tids)
                        self.uid2tids[uid].append(tid)
                        self.tid2uid[tid] = uid
            
            frame2uid = defaultdict(dict)
            uid2frame = defaultdict(list)
            uid2category = dict()
            uid2clipemb = defaultdict(list)
            uid2clip = dict()
            for frame in self.frame2trackid:
                for tid in self.frame2trackid[frame]:
                    frame2uid[frame][self.tid2uid[tid]] = self.frame2trackid[frame][tid]
            for uid in self.uid2tids:
                tids = self.uid2tids[uid]
                for tid in tids:
                    uid2frame[uid] += self.trackid2frame[tid]
                    uid2clipemb[uid].append(self.trackid2clip_emb[tid])
            
            for uid in uid2clipemb:
                emb = torch.stack(uid2clipemb[uid], dim=0)
                emb = torch.mean(emb, dim=0)
                uid2clip[uid] = emb
            save_file = os.path.join(video_dir, 'uid2clip.pkl')
            with open(save_file, 'wb') as f:
                pickle.dump(uid2clip, f)
            
            reid_file = os.path.join(video_dir, 'reid.pkl')
            for uid in self.uid2tids:
                uid2category[uid] = self.trackid2category[self.uid2tids[uid][0]]
            with open(reid_file, 'wb') as f:
                pickle.dump([frame2uid, uid2frame, uid2category], f)


    def replay(self):
        for video_path in self.video_path_list:
            base_name = os.path.basename(video_path).replace(".mp4", "")
            video_dir = os.path.join(self.base_dir, base_name)
            with open(os.path.join(video_dir, 'reid.pkl'), 'rb') as f:
                content = pickle.load(f)
            frame2uid, uid2frame, uid2category = content[0], content[1], content[2]
            cap = cv2.VideoCapture(video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            frame_idx = -1
            writer = imageio.get_writer(os.path.join(video_dir, 'reid.mp4'), fps=15)
            while True:
                success, frame = cap.read()
                frame_idx += 1
                if not success:
                    break
                if frame_idx in frame2uid:
                    for uid in frame2uid[frame_idx]:
                        c = hash_color(uid)
                        x, y, w, h = frame2uid[frame_idx][uid][1]
                        left_top = (int(x-w/2), int(y-h/2))
                        right_bottom = (int(x+w/2), int(y+h/2))
                        cv2.rectangle(frame, left_top, right_bottom, c, 2)
                        label = f'ID: {uid}'
                        label_position = (int(x-w/2)+2, int(y-h/2)+12)
                        cv2.putText(frame, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
                    #cv2.imshow("reid", frame)
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    writer.append_data(frame)
            writer.close()
            cap.release()
            cv2.destroyAllWindows()


    def run(self):
        self.reid_for_all_videos()
        self.replay()
