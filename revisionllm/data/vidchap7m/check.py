import io
import os

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

try:
    from torchvision.transforms import InterpolationMode

    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    from PIL import Image

    BICUBIC = Image.BICUBIC

    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--feat_out_folder", type=str, default='/nfs/data3/user/revisionllm/chapters_clipl14_features')
    parser.add_argument("--data_path", type=str, default="/nfs/data3/user/AllChapters/train.json")

    args = parser.parse_args()
    return args

# --data_path /home/atuin/v100dd/v100dd11/ego4d_data/v2/naq_datasets/train_v1.jsonl --video_folder /home/atuin/v100dd/v100dd11/ego4d_data/v2/full_scale --clip_path /home/atuin/v100dd/v100dd11/revisionllm/checkpoints/clip/ViT-L-14.pt --feat_out_folder /home/atuin/v100dd/v100dd11/revisionllm/naq_clipL14_features

if __name__ == "__main__":
    args = parse_args()
    for file in ["/nfs/data3/user/AllChapters/test.json","/nfs/data3/user/AllChapters/train.json"]:
        ego_data = json.load(open(file))
        print(file, len(ego_data))
        count_npy = 0
        count_mp4 = 0
        for i, data in enumerate(tqdm(ego_data)):
            id = data
            path = os.path.join(args.feat_out_folder, data + '.npy')
            if os.path.exists(path):
                count_npy += 1
            path = f"/nfs/data3/user/AllChapters/videos/{data}.mp4"
            if os.path.exists(path):
                count_mp4 += 1
        print('npy: ',count_npy)
        print('mp4:', count_mp4)
    print()



