import io
import math
import os
import random
import lmdb

root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
import sys
sys.path.append(root_dir)

import re
import argparse
import torch
import json
import numpy as np
from tqdm import tqdm
from vtimellm.model.builder import load_pretrained_model
from vtimellm.utils import disable_torch_init
from vtimellm.inference import inference
from vtimellm.model.adapter.tensor_utils import pad_sequences_1d
from vtimellm.eval.similarity import _topk_pooling
from vtimellm.uncertainty.funs_get_feature_X import get_entropy_statistics
random.seed(42)
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    from PIL import Image
    BICUBIC = Image.BICUBIC



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_path", type=str, default="/home/stud/hannan/VTimeLLM/checkpoints/clip/ViT-L-14.pt")
    parser.add_argument("--model_base", type=str, default="/home/stud/hannan/.cache/huggingface/hub/models--lmsys--vicuna-7b-v1.5/snapshots/de56c35b1763eaae20f4d60efd64af0a9091ebe5")
    parser.add_argument("--pretrain_mm_mlp_adapter", type=str, default=None)#"/home/stud/hannan/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin")
    parser.add_argument("--pretrain_clip_adapter", type=str, default=None)#"/home/stud/hannan/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin")
    parser.add_argument("--stage2", type=str, default='/home/stud/hannan/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage2')
    parser.add_argument("--stage3", type=str, default=None)
    parser.add_argument("--data_path", type=str, default='/nfs/data3/hannan/mad_data_for_cone/data/mad_ori_data/MAD_val.json')
    parser.add_argument("--feat_folder", type=str, default='/nfs/data3/hannan/mad_data_for_cone/offline_lmdb/CLIP_L14_frames_features_5fps')
    parser.add_argument("--task", type=str, default='grounding', choices=['all', 'grounding', 'captioning'])
    parser.add_argument("--log_path", type=str, default='/home/stud/hannan/VTimeLLM/checkpoints/vtimellm-vicuna-v1-5-7b-stage2')
    parser.add_argument("--debug_window", type=int, default=125)
    parser.add_argument("--num_frames", type=int, default=250)
    parser.add_argument("--mlp_adapter", type=bool, default=False)
    parser.add_argument("--ca_adapter", type=bool, default=False)
    parser.add_argument("--cross_attn", type=bool, default=False)
    parser.add_argument("--q_feat_dir", type=str, default=None)
    parser.add_argument("--max_seq_length", type=int, default=2048)
    parser.add_argument("--self_attn", type=str, default=None)
    parser.add_argument("--ca_self_attn", type=str, default=None)
    parser.add_argument("--sa_pos", type=int, default=1)
    parser.add_argument("--neg_window", type=bool, default=False)
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--split", type=int, default=0)
    parser.add_argument("--total_split", type=int, default=1)
    parser.add_argument("--topk_pool", type=bool, default=True)
    parser.add_argument("--adapter_input_dim", type=int, default=768)
    parser.add_argument("--feature_fps", type=float, default=5)
    parser.add_argument("--load_ckp", type=bool, default=False)
    parser.add_argument("--mad_prompt", type=str, default='mad_grounding')
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--clip_adapter", type=bool, default=False)
    parser.add_argument("--clip_adapter_text", type=bool, default=False)
    parser.add_argument("--vis_feat_storage", type=str, default='lmdb', choices=['lmdb', 'npy', 'pth'])
    parser.add_argument("--score", type=str, default='mean_entropy', choices=['cosine_sim', 'max_entropy', 'mean_entropy'])
    parser.add_argument("--clip_adapter_feature", type=str, default='temporal') #'cls', "all", "alternate"
    parser.add_argument("--hierarchy", type=bool, default=False)
    parser.add_argument("--score_merge", type=str, default='multiply', choices=['add', 'multiply'])
    parser.add_argument("--normalize", type=bool, default=True)
    parser.add_argument("--skip_small_videos", type=bool, default=True)
    parser.add_argument("--baseline", type=bool, default=False)
    parser.add_argument("--plus_baseline", type=bool, default=False)
    args = parser.parse_args()
    return args

def iou(outputs, gt, num_frames_clip, num_frames_video, scores, plus_baseline):
    # return frames, ious
    frames = []
    filter_scores = []
    clip_frames = {}
    for i, output in enumerate(outputs):
        if plus_baseline and i==len(outputs)-1:
            i=0
        matches = re.search(r"(\d+) (to|and) (\d+)", output)
        if matches:
            from_number = float(matches.group(1))
            to_number = float(matches.group(3))
            if from_number==num_frames_clip-1 and to_number==num_frames_clip-1:
                continue
            if from_number==to_number:
                from_number = max(0, from_number-1)
                to_number = min(num_frames_video, to_number+1)
            clip_frames[i] = (int(from_number),int(to_number))
            from_number = int(i * num_frames_clip // 2 + from_number)
            to_number = int(i * num_frames_clip // 2 + to_number)
            frames.append((from_number,to_number))
            if len(scores)>0:
                filter_scores.append(scores[i])

    s, e = gt
    ious = []
    for frame in frames:
        f, t = frame[0]/num_frames_video, frame[1]/num_frames_video
        intersection = max(0, min(t, e) - max(f, s))
        union = max(t, e) - min(f, s)
        iou = round(intersection / union, 2)
        ious.append(iou)

    return clip_frames, ious, filter_scores  # return frames, iou


def write_log(log_path, video_id, task, query_id, answer, info=None):
    log = {
        'video_id': video_id,
        'task': task,
        'query_id': query_id,
        'answer': answer
    }
    if info is not None:
        log['info'] = info
    with open(log_path, 'a') as f:
        f.write(json.dumps(log) + '\n')

questions = {
    'mad_grounding': 'During which frames can we see {}?',
    'ego_assertive': 'During which frames {}?',
    'ego_question': 'Find the start and end time of the Query from the Video.\nQuery: {}',
    'captioning': ['Could you please describe the events in the video in detail? Be specific about the activities of individuals, their surroundings, and interactions with others. The output should be in JSON format, structured as follows: {"event": "xx", "timestamps": "from xx to xx"}.']
}


def eval(args):
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)
    prediction_path = args.log_path + f'/predictions_streaming_{str(args.split)}.txt'
    print('prediction_path: ', prediction_path)
    disable_torch_init()
    print('torch.cuda.is_available(): ', torch.cuda.is_available())
    # tokenizer, model, context_len = None, None, None
    tokenizer, model, context_len = load_pretrained_model(args, args.stage2, args.stage3, load_ckp=args.load_ckp)
    if torch.cuda.is_available():
        model = model.bfloat16().cuda()
        # model.to(torch.float16)
    else:
        model.to(torch.float32)

    if args.vis_feat_storage=='lmdb': #because chapters data is enormous I use npy directly for now
        appearance_visual_env = lmdb.open(args.feat_folder, readonly=True, create=False, max_readers=4096 * 8, readahead=False)
        appearance_visual_txn = appearance_visual_env.begin(buffers=True)
    if args.q_feat_dir is not None:
        textual_env = lmdb.open(args.q_feat_dir, readonly=True, create=False, max_readers=4096 * 8, readahead=False)
        textual_txn = textual_env.begin(buffers=True)
    errors=[]
    logs = []
    if os.path.exists(prediction_path):
        with open(prediction_path) as f:
            for line in f:
                try:
                    json_data = json.loads(line)
                    logs.append(json_data['query_id'])
                except Exception as e:
                    print(e, line)
    i=0
    if 'jsonl' in args.data_path:
        with open(args.data_path) as f:
            js = [json.loads(line) for line in f]
            js = [(k['query_id'], k) for k in js]
    else:
        js = json.load(open(args.data_path))
        if 'videos' in js:
            js = js['videos']
            js = [(k['query'], k) for k in js]
        else: #MAD_train.json
            js = [(id, item ) for id, item in js.items()]
    # features = torch.zeros((args.num_frames, 768), dtype=torch.float16)
    bin = len(js) // args.total_split
    items = js[args.split * bin:] if args.split == args.total_split - 1 else js[args.split * bin: (args.split + 1) * bin]
    batch = args.batch
    print('batch: ', batch)
    for item in tqdm(items):
        # batch+=1
        # print('batch: ', batch)
        id, data = item
        i += 1
        if id in logs:
            continue
        try:
            movie = data['movie'] if 'movie' in data else data['clip_id']
            movie = data['query_id'] if 'val_frame' in args.feat_folder else movie
            if args.vis_feat_storage == 'lmdb':  # because chapters data is enormous I use npy directly for now
                dump = appearance_visual_txn.get(movie.encode())
                with io.BytesIO(dump) as reader:
                    img_dump = np.load(reader, allow_pickle=True)
                    features = img_dump['features']
            else:
                path = os.path.join(args.feat_folder, movie + '.npy')
                features = np.load(path)

            query_feats = None
            query_cls_feats = None
            if args.q_feat_dir is not None:
                dump = textual_txn.get(id.encode())
                with io.BytesIO(dump) as reader:
                    q_dump = np.load(reader, allow_pickle=True)
                    query_feats = q_dump['token_features']
                    query_cls_feats = q_dump['cls_features']

            gt_len = math.ceil(data['timestamps'][1] - data['timestamps'][0])

            if 'movie_duration' in data and data['movie_duration'] <= args.debug_window:
                if args.skip_small_videos:
                    continue
                else:
                    sampled_indices = np.linspace(0, features.shape[0]-1, args.num_frames, dtype=np.int32)
                    features = features[sampled_indices]

            if args.baseline:
                sampled_indices = np.linspace(0, features.shape[0]-1, int(args.debug_window * args.feature_fps), dtype=np.int32)
                features = features[sampled_indices]

            ctx_l = len(features)
            assert ctx_l > 0, ctx_l
            clip_length = args.debug_window * args.feature_fps
            num_window = math.ceil(ctx_l / (clip_length//2)) - 1
            windowidx = [1] if args.baseline else list(range(num_window))
            clip_feats = []
            for i in windowidx:
                start = max(i * clip_length//2, 0)
                end = min(i * clip_length//2 + clip_length, ctx_l-1)
                sampled_indices = np.linspace(start, end, args.num_frames, dtype=np.int32)
                clip_feat = features[sampled_indices]
                clip_feats.append(clip_feat)

            if args.plus_baseline:
                sampled_indices = np.linspace(0, features.shape[0]-1, args.num_frames, dtype=np.int32)
                clip_feat = features[sampled_indices]
                clip_feats.append(clip_feat)

            features = torch.from_numpy(np.array(clip_feats))
            timestamps = data['timestamps']#[data['timestamps'][0] - start_s, data['timestamps'][1] - start_s]
            if args.q_feat_dir is not None:
                query_feats = torch.from_numpy(query_feats)
                query_cls_feats = torch.from_numpy(query_cls_feats)

            if torch.cuda.is_available():
                features = features.bfloat16().to('cuda')#.to(torch.float16).to('cuda')
                if args.q_feat_dir is not None:
                    query_feats = query_feats.bfloat16().to('cuda')  # .to(torch.float16).to('cuda')
                    query_cls_feats = query_cls_feats.bfloat16().to('cuda')
            else:
                features = features.to(torch.float32)
                if args.q_feat_dir is not None:
                    query_feats = query_feats.to(torch.float32)
                    query_cls_feats = query_cls_feats.to(torch.float32)
            # if args.q_feat_dir is not None:
            #     query_feats = pad_sequences_1d(query_feats[None,].repeat(features.shape[0],1,1), dtype=query_feats.dtype, device=query_feats.device, fixed_length=None)

            if features is None:
                print(f'Can not find video {movie}')
                continue

            if args.task in ['captioning', 'all']:
                for query_id, query in enumerate(questions['captioning']):
                    answer = inference(model, features, query_feats, "<video>\n " + query, tokenizer)
                    write_log(prediction_path, movie, 'captioning', id, answer)

            if args.task in ['grounding', 'all']:
                #for sentence_id, (timestamps, sentence) in enumerate(zip(data['timestamps'], data['sentences'])):
                sentence = data['sentence'].strip().lower() if 'sentence' in data else data['query'].strip('.?').lower()
                if 'sentence' in data and sentence.endswith("."):
                    sentence = sentence[:-1]
                # sentence = data['query']
                # for query_id, query in enumerate(questions['grounding']):
                query = questions[args.mad_prompt]
                answers = []
                score_cos = []
                scores_entropy = []
                for i in range(math.ceil(features.shape[0]/batch)):
                    start = i * batch
                    end = min(start + batch, features.shape[0])
                    feat = features[start:end]
                    if args.q_feat_dir is not None:# and i==0:
                        query_feats_temp = pad_sequences_1d(query_feats[None,].repeat(feat.shape[0],1,1), dtype=query_feats.dtype, device=query_feats.device, fixed_length=None)
                    answer, model_output = inference(model, feat, query_feats_temp, "<video>\n" + query.format(sentence), tokenizer, return_list=True)
                    # print("answer: ", answer)
                    # for a in answer:
                    answers.extend(answer)
                    if args.score=='max_entropy':
                        entropy = get_entropy_statistics(torch.cat([a[:, None] for a in model_output['scores']], 1), 0,
                                                         model_output['scores'][0].shape[1])
                        scores_entropy.extend([e[0].item() for e in entropy])
                    elif args.score=='mean_entropy':
                        entropy = get_entropy_statistics(torch.cat([a[:, None] for a in model_output['scores']], 1), 0,
                                                         model_output['scores'][0].shape[1])
                        scores_entropy.extend([e[2].item() for e in entropy])
                    #break
                # if 'entropy' in args.score:
                #     eps = 0.00001
                #     max_entropy = log(32000, 2)
                #     scores_ = [-s / max_entropy + eps for s in scores]
                duration = data['movie_duration'] if 'movie_duration' in data else data['duration']
                gt = (timestamps[0] / duration, timestamps[1] / duration)
                num_frames_video = int(duration * args.num_frames / args.debug_window)
                frames, ious, scores_entropy = iou(answers, gt, args.num_frames, num_frames_video, scores_entropy, args.plus_baseline)
                # if args.score == 'cosine_sim':
                for k,v in frames.items():
                    proposal_feat = features[k][v[0]:v[1]+1]
                    proposal_features = proposal_feat / proposal_feat.norm(dim=0, keepdim=True)
                    if args.topk_pool:
                        proposal_features = _topk_pooling(query_cls_feats[None], proposal_features[None], min(proposal_features.shape[0], 3))[0]
                        score = torch.einsum('bd,d->b', proposal_features, query_cls_feats)
                    else:
                        score = torch.einsum('bd,d->b', proposal_features, query_cls_feats).mean()
                    # if 'entropy' in args.score:
                    #     scores[i] *= score.item()
                    # else:
                    score_cos.append(score.item())
                if args.normalize:
                    if len(score_cos)>0:
                        m_s = max(score_cos)
                        score_cos = [e/m_s for e in score_cos]
                    if len(scores_entropy)>0:
                        m_s = max(scores_entropy)
                        scores_entropy = [e/m_s for e in scores_entropy]
                if 'entropy' in args.score:
                    if args.score_merge == 'add':
                        scores = [a - b for a, b in zip(score_cos, scores_entropy)]
                    elif args.score_merge == 'multiply':
                        scores = [a / b for a, b in zip(score_cos, scores_entropy)]
                    else:
                        scores = [-e for e in scores_entropy]
                else:
                    scores = score_cos
                write_log(prediction_path, movie, 'grounding', id, answers, info={'iou': ious, 'scores': scores})
        except:
            # if args.debug:
            #     raise
            errors.append(id)
    print('errors', errors)

def calculate_result(args):
    logs = []
    for i in range(args.total_split):
        prediction_path = args.log_path + f'/predictions_stream_{str(i)}.txt'
        with open(prediction_path) as f:
            for line in f:
                try:
                    json_data = json.loads(line)
                    logs.append(json_data)
                except Exception as e:
                    print(e, line)
    ious = [x['info']['iou'] for x in logs if x['task'] == 'grounding' and x['info']['iou']!=-1]
    fn = [x['info']['fn'] for x in logs if x['task'] == 'grounding' and 'fn' in x['info']]
    fp = [x['info']['fp'] for x in logs if x['task'] == 'grounding' and 'fp' in x['info']]
    l = len(ious)
    lf = len(fp) // 2
    print(f"Found {l} logs")
    metrics = {
        "mIoU": sum(ious) / l * 100,
        "fn": sum(fn) / lf * 100,
        "fp": sum(fp) / lf * 100
    }
    for m in [0.1, 0.3, 0.5, 0.7]:
        metrics[f"R1@{m}"] = sum(iou >= m for iou in ious) / l * 100
    return metrics


if __name__ == "__main__":
    args = parse_args()
    eval(args)