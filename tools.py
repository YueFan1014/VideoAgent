import os.path as osp
import json
import pickle
from encoder import encode_sentences
from utils import compute_cosine_similarity, top_k_indices
import os
import cv2
from database import DataBase
from moviepy.editor import VideoFileClip
import socket
import sys
from io import StringIO
import torch
from InternVid.viclip import get_viclip, retrieve_text, _frame_from_video, frames2tensor, get_vid_feat, get_text_feat_dict
import requests
import numpy as np
import base64
import gc


model_cfgs = {
    'viclip-l-internvid-10m-flt': {
        'size': 'l',
        'pretrained': 'tool_models/viCLIP/ViClip-InternVid-10M-FLT.pth',
    }

}          


class ToolKit:
    def __init__(self, video_path, base_dir='preprocess', vqa_tool='videollava', use_reid=True, openai_api_key='your_openai_api_key'):
        self.video_path = video_path
        base_name = os.path.basename(video_path).replace(".mp4", "")
        self.video_dir = os.path.join(base_dir, base_name)
        assert vqa_tool in ["videollava", "gpt-4v"]
        self.vqa_tool = vqa_tool
        cap = cv2.VideoCapture(video_path)
        self.fps = round(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        self.openai_api_key = openai_api_key

        
        with open(osp.join(self.video_dir, 'captions.json')) as f:
            captions = json.load(f)
        self.segments = list(captions.keys())
        self.captions = list(captions.values())
        self.segment_num = len(self.segments)
        self.database = DataBase(video_path, base_dir=base_dir, use_reid=use_reid)
        
        cfg = model_cfgs['viclip-l-internvid-10m-flt']
        model = get_viclip(cfg['size'], cfg['pretrained'])
        assert(type(model)==dict and model['viclip'] is not None and model['tokenizer'] is not None)
        self.viclip, self.tokenizer = model['viclip'], model['tokenizer']
        self.viclip = self.viclip.to("cuda")


    def query_database(self, program):
        res = self.database.query_database(program=program)
        return str(res)
    

    def retrieve_candidate_objects(self, description):
        res = self.database.retrieve_candidate_objects(description=description)
        return str(res)


    def caption_retrieval(self, start_segment, end_segment):
        d = f"There are {self.segment_num} segments in total, ranging from 0 to {self.segment_num-1}. "
        start_segment = max(start_segment, 0)
        end_segment = min(end_segment, self.segment_num-1)
        if start_segment > end_segment:
            return d+"Invalid start and end segment IDs!"
        res = dict()
        for segment_id in range(start_segment, end_segment+1):
            res[segment_id] = self.captions[segment_id]
        return d+str(res)


    def segment_localization(self, description, k=5):
        with open(osp.join(self.video_dir, 'segment_textual_embedding.pkl'), 'rb') as f:
            segment2textual_emb = pickle.load(f)
        des2textual_emb = encode_sentences(sentence_list=[description], model_name='text-embedding-3-large')
        textual_scores = compute_cosine_similarity(target_embedding=des2textual_emb, embedding_list=segment2textual_emb)
        
        with open(osp.join(self.video_dir, 'segment_visual_embedding.pkl'), 'rb') as f:
            segment2visual_emb = pickle.load(f)
        with torch.no_grad():
            des2visual_emb = self.viclip.get_text_features(description, self.tokenizer, {}).cpu().numpy()
        #print(des2visual_emb.shape)
        visual_scores = compute_cosine_similarity(target_embedding=des2visual_emb, embedding_list=segment2visual_emb)
        #print(visual_scores)
        #print(textual_scores)
        ensemble_scores = 18*visual_scores +11*textual_scores
        k_indices = top_k_indices(ensemble_scores, k)
        candidate_segment2caption = dict()
        for idx in k_indices:
            candidate_segment2caption[idx] = self.captions[idx]
        res = f"There are {self.segment_num} segments in total, ranging from 0 to {self.segment_num-1}. The most relevant segments are: {candidate_segment2caption}."
        return res
    

    def videollava_VQA(self, question, segment_id):
        #print(segment_id, self.segment_num)
        if segment_id not in range(self.segment_num):
            return f"Segment ID {segment_id} not in range 0-{self.segment_num-1}."
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        client.connect("tmp/vqa.sock")
        content_file = 'tmp/content.pkl'
        video_segment_path = osp.join(self.video_dir, f'segment_{segment_id}.mp4')
        if not osp.exists(video_segment_path):
            #create segment mp4 video
            candidate_segments = []
            for i in range(segment_id-1, segment_id+2):
                if i < 0 or i > self.segment_num-1:
                    continue
                candidate_segments.append(i)
            segment_start_second = candidate_segments[0]*2
            segment_end_second = (candidate_segments[-1]+1)*2

            original_stdout = sys.stdout
            output_catcher = StringIO()
            sys.stdout = output_catcher
            video = VideoFileClip(self.video_path)
            segment_video = video.subclip(segment_start_second, segment_end_second)
            segment_video.write_videofile(video_segment_path)
            sys.stdout = original_stdout

        content = dict()
        content['video_path'] = video_segment_path
        content['question'] = question
        with open(content_file, 'wb') as f:
            pickle.dump(content, f)
        client.send(b'sent')
        res = client.recv(1024).decode('utf-8')
        with open(content_file, 'rb') as f:
            ans = pickle.load(f)
        client.send(b'finish')
        return ans
    

    def gpt4v_VQA(self, question, segment_id):
        if segment_id not in range(self.segment_num):
            return f"Segment ID {segment_id} not in range 0-{self.segment_num-1}."
        candidate_segments = []
        for i in range(segment_id-1, segment_id+2):
            if i < 0 or i > self.segment_num-1:
                continue
            candidate_segments.append(i)
        image_start_frame = candidate_segments[0]*self.fps * 2
        image_end_frame = (candidate_segments[-1]+1)*self.fps * 2
        frame_interval = (image_end_frame-image_start_frame)//3
        cap = cv2.VideoCapture(self.video_path)
        target_frame_ids = []
        for i in range(4):
            target_frame_id = image_start_frame+i*frame_interval
            target_frame_ids.append(target_frame_id)
            cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame_id)
            success, frame = cap.read()
            if not success:
                frame = np.zeros([24, 24, 3])
            image_path = osp.join(self.video_dir, f'frame_{target_frame_id}.jpg')
            cv2.imwrite(image_path, frame)
        def encode_image(image_path):
            with open(image_path, "rb") as image_file:
                tmp = image_file.read()
            return base64.b64encode(tmp).decode('utf-8')
        # Getting the base64 string
        base64_images = []
        for id in target_frame_ids:
            image_path = osp.join(self.video_dir, f'frame_{id}.jpg')
            base64_images.append(encode_image(image_path))

        headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {self.openai_api_key}"
        }
        question = f"The images are sequential frames in a 6-second video. Please briefly describe what happened in the video then briefly provide the answer to the question '{question}'."
        payload = {
        "model": "gpt-4-vision-preview",
        "messages": [
            {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": question
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_images[0]}",},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_images[1]}",},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_images[2]}",},
                },
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{base64_images[3]}",},
                },
            ]
            }
        ],
        "max_tokens": 200
        }
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        return response.json()['choices'][0]['message']['content']
    
    
    def visual_question_answering(self, question, segment_id):
        if self.vqa_tool == 'videollava':
            return self.videollava_VQA(question=question, segment_id=segment_id)
        elif self.vqa_tool == 'gpt-4v':
            return self.gpt4v_VQA(question=question, segment_id=segment_id)

