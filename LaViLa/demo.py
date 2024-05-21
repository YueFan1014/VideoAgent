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


video_path = 'assets/3c0dffd0-e38e-4643-bc48-d513943dc20b_012_014.mp4'


from base64 import b64encode


# The video is represented by `num_seg=4` frames
vr = decord.VideoReader(video_path)
print("total length:", len(vr))
num_seg = 4
frame_ids = get_frame_ids(0, len(vr), num_segments=num_seg, jitter=False)
frames = video_loader_by_frames('./', video_path, frame_ids)
print(frames)
print('frames_size:', frames.size()) #[num_seg, w, h, 3]


# display the subsampled frames
# plt.figure(figsize=(16, 40))
# for i in range(num_seg):
#   plt.subplot(1, num_seg, i + 1)
#   plt.imshow(frames[i].cpu().numpy().astype(int))
#   plt.axis('off')
# plt.show()


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

num_params = sum(p.numel() for p in model.parameters())
print(f'model params: {num_params}')
model.eval()
#model.cuda()
print('loaded into GPU')
# transforms on input frames
crop_size = 224
val_transform = transforms.Compose([
    Permute([3, 0, 1, 2]),
    transforms.Resize(crop_size),
    transforms.CenterCrop(crop_size),
    transforms_video.NormalizeVideo(mean=[108.3272985, 116.7460125, 104.09373615000001], std=[68.5005327, 66.6321579, 70.32316305])
])
frames = val_transform(frames) 
print("frames shape before squeeze: ", frames.size()) #[3, 4, 224, 224]
frames = frames.unsqueeze(0) # fake a batch dimension
print("frames shape: ", frames.size()) #[1, 3, 4, 224, 224]

tokenizer = MyGPT2Tokenizer('gpt2', add_bos=True)

candidate_num = 5

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



start_time = time.time()
for i in range(100):
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
            temperature=0.7,
            early_stopping=True,
        )
        for i in range(candidate_num):
            generated_text_str = decode_one(generated_text_ids[i], tokenizer)
            print('{}: {}'.format(i, generated_text_str))
end_time = time.time()
print(end_time-start_time)