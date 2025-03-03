import io
import os

root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..")
import sys

sys.path.append(root_dir)

import math
import clip
import re
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
import decord
from csv import DictReader
import glob
import os.path

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
    parser.add_argument("--bsz", type=int, default=5)
    parser.add_argument("--clip_path", type=str, default="/home/stud/user/VTimeLLM/checkpoints/clip/ViT-L-14.pt")
    parser.add_argument("--data_path", type=str, default='/nfs/data8/user/msrvtt/MSRVTT/videos/all/')#"/nfs/data3/user/ego4d_data/v2/annotations/nlq_val.json")
    parser.add_argument("--feat_out_folder", type=str, default='/nfs/data8/user/msrvtt/MSRVTT/videos/features/')
    parser.add_argument("--video_folder", type=str, default="/nfs/data8/user/msrvtt/MSRVTT/videos/all/")
    parser.add_argument("--naq", type=bool, default=True)

    args = parser.parse_args()
    return args

# --data_path /home/atuin/v100dd/v100dd11/ego4d_data/v2/annotations/nlq_train.json
# --video_folder /home/atuin/v100dd/v100dd11/ego4d_data/v2/down_scale_nlq
# --clip_path /home/atuin/v100dd/v100dd11/revisionllm/checkpoints/clip/ViT-L-14.pt
# --feat_out_folder /home/atuin/v100dd/v100dd11/revisionllm/ego4g_clip_features

# --data_path /home/atuin/v100dd/v100dd11/ego4d_data/v2/naq_datasets/train_v1.jsonl
# --video_folder /home/atuin/v100dd/v100dd11/ego4d_data/v2/down_scale_nlq
# --clip_path /home/atuin/v100dd/v100dd11/revisionllm/checkpoints/clip/ViT-L-14.pt
# --feat_out_folder /home/atuin/v100dd/v100dd11/revisionllm/naq_clip_features


if __name__ == "__main__":
    args = parse_args()
    disable_torch_init()
    ego_data = glob.glob(os.path.join(args.data_path, '*.mp4'))

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
    bin = len(ego_data) // args.total_split
    ego_data = ego_data[args.split * bin:] if args.split == args.total_split - 1 else ego_data[args.split * bin: (args.split + 1) * bin]
    for data in tqdm(ego_data):
        if True:#try:
            features = None
            id = data.split('/')[-1].split('.')[0] #data['video_id'] if args.naq else data['video_uid']
            start_end = None
            path = os.path.join(args.feat_out_folder, id + '.npy')
            if os.path.exists(path):
                continue
            for ext in ['mp4', 'mkv', 'webm']:
                video_path = data #os.path.join(args.video_folder, f"{id}.{ext}")
                if os.path.isfile(video_path):
                    print(id)
                    features = []
                    ind = []
                    video_reader = decord.VideoReader(video_path, num_threads=1)
                    video_reader.skip_frames(1)
                    start = 0  # * video_reader.get_avg_fps())
                    end = len(video_reader)-1  # min(int(start_end[1] * video_reader.get_avg_fps()), len(video_reader)) - 1
                    total_frames = end - start + 1
                    sample_fps = 5
                    sampled_indices = np.linspace(start, end, total_frames, dtype=np.int32) #int((total_frames * sample_fps) // video_reader.get_avg_fps())
                    chunk = len(sampled_indices) // args.bsz
                    for i in range(args.bsz):
                        #_, images, sampled_indices = video_loader.extract({'id': None, 'video': video_path}, se, 2)
                        ####################################
                        indices = sampled_indices[i * chunk:] if i == args.bsz - 1 else sampled_indices[i*chunk:(i+1)*chunk]
                        ind.extend(indices)
                        sampled_frames = video_reader.get_batch(indices).asnumpy()
                        images = torch.from_numpy(sampled_frames.transpose((0, 3, 1, 2)))
                        # print("load success")
                        images = transform(images / 255.0)
                        images = images.to(torch.float32)#images.to(torch.float16) if torch.cuda.is_available() else images.to(torch.float32)
                        with torch.no_grad():
                            feature = clip_model.encode_image(images.to('cuda') if torch.cuda.is_available() else images)
                            features.append(feature)
                    features = torch.cat(features)
                    # print("extraction success")

            if features is None:
                print(f'Can not find video {id}')
                continue
            np.save(path, np.array(features.detach().cpu()).astype(np.float32))
            # print("save success")
        #except Exception as e:
            #print(e)