import glob
import os
import random

root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
import sys

sys.path.append(root_dir)

import clip
import re
import math
import argparse
import torch
import json
import numpy as np
from tqdm import tqdm
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
from revisionllm.model.builder import load_pretrained_model
from revisionllm.utils import disable_torch_init
from revisionllm.mm_utils import VideoExtractor
from revisionllm.inference import inference
import subprocess

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    from PIL import Image

    BICUBIC = Image.BICUBIC

    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--total_split", type=int, default=1)
    parser.add_argument("--clip_path", type=str, default="/checkpoints/clip/ViT-L-14.pt")
    parser.add_argument("--data_path", type=str, default="/data/chapters/train.json")
    parser.add_argument("--feat_out_folder", type=str, default='/data/chapters/chapters_clipl14_features')
    parser.add_argument("--video_folder", type=str, default="/data/chapters/videos")
    parser.add_argument("--batch", type=int, default=1000)
    parser.add_argument("--random", type=bool, default=True)
    args = parser.parse_args()
    return args

# --data_path pathego4d_data/v2/naq_datasets/train_v1.jsonl --video_folder pathego4d_data/v2/full_scale --clip_path pathvtimellm/checkpoints/clip/ViT-L-14.pt --feat_out_folder pathvtimellm/naq_clipL14_features

if __name__ == "__main__":
    args = parse_args()
    disable_torch_init()
    ego_data = json.load(open(args.data_path))
    print('torch.cuda.is_available(): ', torch.cuda.is_available())
    if not os.path.exists(args.feat_out_folder):
        os.makedirs(args.feat_out_folder)

    clip_model, _ = clip.load(args.clip_path)
    clip_model.eval()
    if torch.cuda.is_available():
        clip_model = clip_model.cuda()

    video_loader = VideoExtractor(N=100)

    transform = Compose([
        Resize(224, interpolation=BICUBIC),
        CenterCrop(224),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ])
    path_list = []
    images_list = []
    files1 = glob.glob(os.path.join(args.video_folder, '*.mp4'))
    # existing_files = glob.glob(os.path.join(args.feat_out_folder, '*.npy'))
    # existing_files = [e.split('/')[-1].split('.')[0] for e  in existing_files]
    ego_data = files1
    bin = len(ego_data) // args.total_split
    ego_data = ego_data[args.split * bin:] if args.split == args.total_split - 1 else ego_data[args.split * bin: (args.split + 1) * bin]
    random.seed(args.split)
    if args.random:
        random.shuffle(ego_data)
    for i, data in enumerate(tqdm(ego_data)):
        try:
            features = None
            id = data
            path = os.path.join(args.feat_out_folder, data.split('/')[-1].split('.')[0] + '.npy')
            # if os.path.exists(path):
            #     os.remove(data)
            #     continue
            #start_end = [data['timestamps'][0], data['timestamps'][1]]
            ext = 'mp4'#, 'mkv', 'webm']:
            video_path = data#os.path.join(args.video_folder, f"{id}.{ext}")
            if os.path.isfile(video_path):
                # new_videos.append(id)
                _, images, _ = video_loader.extract({'id': None, 'video': video_path}, None, 2)
                images = transform(images / 255.0)
                images = images.to(torch.float16) if torch.cuda.is_available() else images.to(torch.float32)

                with torch.no_grad():
                    n_frames = len(images)
                    n_batch = int(math.ceil(n_frames / args.batch))
                    video_features = []
                    for i in range(n_batch):
                        st_idx = i * args.batch
                        ed_idx = (i + 1) * args.batch
                        _video_frames = images[st_idx:ed_idx]
                        _video_features = clip_model.encode_image(_video_frames.to('cuda') if torch.cuda.is_available() else images)
                        video_features.append(_video_features)
                    video_features = torch.cat(video_features, dim=0)
                    np.save(path, np.array(video_features.detach().cpu()).astype(np.float32))
                    # command = f'rsync -avz -r --info=progress2 --info=name0 --ignore-existing  {path} path@csnhr.nhr.fau.de:pathvtimellm/chapters_clipl14_features/chapters_clipl14_features'
                    print("$$$$$$$$ SUCCESS $$$$$$$$ ", id)
                    # process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)
                    # output, error = process.communicate()
                os.remove(data)
                break
        except Exception as e:
            print("$$$$$$$$ ERROR $$$$$$$$ ", id)
            if os.path.exists(data):
                os.remove(data)
            print(e)
