import decord
import matplotlib.pyplot as plt
import numpy as np
from collections import OrderedDict
import time
import torch
import torchvision.transforms as transforms
import torchvision.transforms._transforms_video as transforms_video
import sys
sys.path.insert(0, './')
from lavila.data.video_transforms import Permute
from lavila.data.datasets import get_frame_ids, video_loader_by_frames
from lavila.models.models import VCLM_OPENAI_TIMESFORMER_BASE_GPT2
from lavila.models.tokenizer import MyGPT2Tokenizer
from base64 import b64encode
import os
import fnmatch
import imageio
import json
import cv2


ckpt_path = 'vclm_openai_timesformer_base_gpt2_base.pt_ego4d.jobid_319630.ep_0002.md5sum_68a71f.pth'
ckpt = torch.load(ckpt_path, map_location='cpu')
state_dict = OrderedDict()
for k, v in ckpt['state_dict'].items():
  state_dict[k.replace('module.', '')] = v

# instantiate the model, and load the pre-trained weights
model = VCLM_OPENAI_TIMESFORMER_BASE_GPT2(
    text_use_cls_token=False,
    project_embed_dim=256,
    gated_xattn=True,
    timesformer_gated_xattn=False,
    freeze_lm_vclm=False,
    freeze_visual_vclm=False,
    freeze_visual_vclm_temporal=False,
    num_frames=4,
    drop_path_rate=0.
)

model.load_state_dict(state_dict, strict=True)
model.eval()
tokenizer = MyGPT2Tokenizer('gpt2', add_bos=True)

candidate_num = 5
crop_size = 224
val_transform = transforms.Compose([
    Permute([3, 0, 1, 2]),
    transforms.Resize(crop_size),
    transforms.CenterCrop(crop_size),
    transforms_video.NormalizeVideo(mean=[108.3272985, 116.7460125, 104.09373615000001], std=[68.5005327, 66.6321579, 70.32316305])
])


def decode_one(generated_ids, tokenizer):
    # get the index of <EOS>
    if tokenizer.eos_token_id == tokenizer.bos_token_id:
        if tokenizer.eos_token_id in generated_ids[1:].tolist():
            eos_id = generated_ids[1:].tolist().index(tokenizer.eos_token_id) + 1
        else:
            eos_id = len(generated_ids.tolist()) - 1
    elif tokenizer.eos_token_id in generated_ids.tolist():
        eos_id = generated_ids.tolist().index(tokenizer.eos_token_id)
    else:
        eos_id = len(generated_ids.tolist()) - 1
    generated_text_str = tokenizer.tokenizer.decode(generated_ids[1:eos_id].tolist())
    return generated_text_str


def create_caption(frames):
    with torch.no_grad():
        image_features = model.encode_image(frames)
        generated_text_ids, ppls = model.generate(
            image_features,
            tokenizer,
            target=None, # free-form generation
            max_text_length=77,
            top_k=None,
            top_p=0.95,  # nucleus sampling
            num_return_sequences=candidate_num, # number of candidates: 10
            temperature=0.9,
            early_stopping=True,
        )
        longest_sentence = ""
        for i in range(candidate_num):
            generated_text_str = decode_one(generated_text_ids[i], tokenizer)
            if len(generated_text_str) > len(longest_sentence):
                longest_sentence = generated_text_str
        return longest_sentence


def captioning(frame_path, fps, caption_seconds=2, frames_per_caption=4):
    frame_interval = int(fps*caption_seconds/frames_per_caption)
    sequential_image_list = []
    sequential_caption_list = dict()

    for root, dirs, files in os.walk(frame_path):
        for file in files:
            if fnmatch.fnmatch(file, '*.jpg'):
                sequential_image_list.append(file)

    sequential_image_list.sort() # ordered frame list

    start_frame = int(sequential_image_list[0].split('.')[0].split('_')[-1])
    end_frame = int(sequential_image_list[-1].split('.')[0].split('_')[-1])

    print(start_frame)
    print(end_frame)
    total_frames = end_frame-start_frame+1

    total_captions = total_frames//(fps*caption_seconds)
    IMAGE_NAME_PATTERN = "video_frame_{:07d}.jpg"


    for i in range(total_captions):
        print(i)
        caption_start_frame = start_frame + i * fps * caption_seconds
        caption_end_frame = start_frame + (i+1) * fps * caption_seconds
        input_frames = []
        for j in range(frames_per_caption):
            frame_idx = caption_start_frame + j* frame_interval
            print('frame: ', frame_idx)
            frame_name = IMAGE_NAME_PATTERN.format(frame_idx)
            image_file = os.path.join(frame_path, frame_name)
            image = imageio.imread(image_file)
            input_frames.append(image)
        input_frames = torch.from_numpy(np.stack(input_frames, axis=0)).float() #[4, w, h, 3]
        #print("input_frames: ", input_frames)
        #print("input_frames.size: ", input_frames.size())
        frames = val_transform(input_frames) 
        frames = frames.unsqueeze(0)
        caption = create_caption(frames)
        time_stamps = "{}-{}".format(str(caption_start_frame), str(caption_end_frame))
        sequential_caption_list[time_stamps] = caption

    with open(os.path.join(frame_path, 'captions.json'), 'w') as f:
        json.dump(sequential_caption_list, f)



def captioning(frame_path, fps, caption_seconds=2, frames_per_caption=4):
    frame_interval = int(fps*caption_seconds/frames_per_caption)
    sequential_image_list = []
    sequential_caption_list = dict()

    for root, dirs, files in os.walk(frame_path):
        for file in files:
            if fnmatch.fnmatch(file, '*.jpg'):
                sequential_image_list.append(file)

    sequential_image_list.sort() # ordered frame list

    start_frame = int(sequential_image_list[0].split('.')[0].split('_')[-1])
    end_frame = int(sequential_image_list[-1].split('.')[0].split('_')[-1])

    print(start_frame)
    print(end_frame)
    total_frames = end_frame-start_frame+1

    total_captions = total_frames//(fps*caption_seconds)
    IMAGE_NAME_PATTERN = "video_frame_{:07d}.jpg"


    for i in range(total_captions):
        print(i)
        caption_start_frame = start_frame + i * fps * caption_seconds
        caption_end_frame = start_frame + (i+1) * fps * caption_seconds
        input_frames = []
        for j in range(frames_per_caption):
            frame_idx = caption_start_frame + j* frame_interval
            print('frame: ', frame_idx)
            frame_name = IMAGE_NAME_PATTERN.format(frame_idx)
            image_file = os.path.join(frame_path, frame_name)
            image = imageio.imread(image_file)
            input_frames.append(image)
        input_frames = torch.from_numpy(np.stack(input_frames, axis=0)).float() #[4, w, h, 3]
        #print("input_frames: ", input_frames)
        #print("input_frames.size: ", input_frames.size())
        frames = val_transform(input_frames) 
        frames = frames.unsqueeze(0)
        caption = create_caption(frames)
        time_stamps = "{}-{}".format(str(caption_start_frame), str(caption_end_frame))
        sequential_caption_list[time_stamps] = caption

    with open(os.path.join(frame_path, 'captions.json'), 'w') as f:
        json.dump(sequential_caption_list, f)