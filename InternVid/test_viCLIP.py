import numpy as np
import os
import cv2

from viclip import get_viclip, retrieve_text, _frame_from_video
video = cv2.VideoCapture('Data/InternVid/example1.mp4')
frames = [x for x in _frame_from_video(video)]
print('frames', frames)
# modify xxx to the path of the pretrained model
model_cfgs = {
    'viclip-l-internvid-10m-flt': {
        'size': 'l',
        'pretrained': '/home/yue/data/ViClip-InternVid-10M-FLT.pth',
    },
    'viclip-l-internvid-200m': {
        'size': 'l',
        'pretrained': 'xxx/ViCLIP-L_InternVid-200M.pth',
    },
    'viclip-b-internvid-10m-flt': {
        'size': 'b',
        'pretrained': 'xxx/ViCLIP-B_InternVid-FLT-10M.pth',
    },
    'viclip-b-internvid-200m': {
        'size': 'b',
        'pretrained': 'xxx/ViCLIP-B_InternVid-200M.pth',
    },
}

text_candidates = ["A playful dog and its owner wrestle in the snowy yard, chasing each other with joyous abandon.",
                   "A man in a gray coat walks through the snowy landscape, pulling a sleigh loaded with toys.",
                   "A person dressed in a blue jacket shovels the snow-covered pavement outside their house.",
                   "A pet dog excitedly runs through the snowy yard, chasing a toy thrown by its owner.",
                   "A person stands on the snowy floor, pushing a sled loaded with blankets, preparing for a fun-filled ride.",
                   "A man in a gray hat and coat walks through the snowy yard, carefully navigating around the trees.",
                   "A playful dog slides down a snowy hill, wagging its tail with delight.",
                   "A person in a blue jacket walks their pet on a leash, enjoying a peaceful winter walk among the trees.",
                   "A man in a gray sweater plays fetch with his dog in the snowy yard, throwing a toy and watching it run.",
                   "A person bundled up in a blanket walks through the snowy landscape, enjoying the serene winter scenery."]

cfg = model_cfgs['viclip-l-internvid-10m-flt']
model_l = get_viclip(cfg['size'], cfg['pretrained'])
print('a')
texts, probs = retrieve_text(frames, text_candidates, models=model_l, topk=5)
