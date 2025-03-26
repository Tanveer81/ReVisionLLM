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
from revisionllm.model.builder import load_pretrained_model
from revisionllm.utils import disable_torch_init
from revisionllm.inference import inference
from revisionllm.model.adapter.tensor_utils import pad_sequences_1d
from revisionllm.eval.similarity import _topk_pooling
from revisionllm.uncertainty.funs_get_feature_X import get_entropy_statistics


random.seed(42)
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    from PIL import Image
    BICUBIC = Image.BICUBIC



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip_path", type=str, default="/home/stud/user/VTimeLLM/checkpoints/clip/ViT-L-14.pt")
    parser.add_argument("--model_base", type=str, default="/home/stud/user/.cache/huggingface/hub/models--lmsys--vicuna-7b-v1.5/snapshots/de56c35b1763eaae20f4d60efd64af0a9091ebe5")
    parser.add_argument("--pretrain_mm_mlp_adapter", type=str, default=None)#"/home/stud/user/VTimeLLM/checkpoints/revisionllm-vicuna-v1-5-7b-stage1/mm_projector.bin")
    parser.add_argument("--pretrain_clip_adapter", type=str, default=None)
    parser.add_argument("--stage2", type=str, default='/home/stud/user/VTimeLLM/checkpoints/revisionllm-vicuna-v1-5-7b-stage2')
    parser.add_argument("--stage3", type=str, default=None)
    parser.add_argument("--data_path", type=str, default='/nfs/data3/user/mad_data_for_cone/data/mad_ori_data/MAD_val.json')
    parser.add_argument("--feat_folder", type=str, default='/nfs/data3/user/mad_data_for_cone/offline_lmdb/CLIP_L14_frames_features_5fps')
    parser.add_argument("--task", type=str, default='grounding', choices=['all', 'grounding', 'captioning'])
    parser.add_argument("--log_path", type=str, default='/home/stud/user/VTimeLLM/checkpoints/revisionllm-vicuna-v1-5-7b-stage2')
    parser.add_argument("--debug_window", type=int, default=125)
    parser.add_argument("--num_frames", type=int, default=250)
    parser.add_argument("--hierarchy_num_videos", type=int, default=33)
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
    parser.add_argument("--topk_pool", type=bool, default=False)
    parser.add_argument("--adapter_input_dim", type=int, default=256)
    parser.add_argument("--feature_fps", type=float, default=5)
    parser.add_argument("--load_ckp", type=bool, default=False)
    parser.add_argument("--mad_prompt", type=str, default='mad_grounding')
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--vis_feat_storage", type=str, default='lmdb', choices=['lmdb', 'npy', 'pth'])
    parser.add_argument("--clip_adapter", type=bool, default=False)
    parser.add_argument("--clip_adapter_text", type=bool, default=False)
    parser.add_argument("--clip_adapter_feature", type=bool, default=False)
    parser.add_argument("--hierarchy", type=bool, default=False)
    parser.add_argument("--score", type=str, default='mean_entropy', choices=['cosine_sim', 'max_entropy', 'mean_entropy'])
    parser.add_argument("--score_merge", type=str, default='multiply', choices=['add', 'multiply'])
    parser.add_argument("--normalize", type=bool, default=True)
    parser.add_argument("--hierarchy_all", type=bool, default=False)
    parser.add_argument("--high_res_log_path", type=str,default=None)
    parser.add_argument("--single", type=bool, default=True)
    parser.add_argument("--zoom", type=int, default=1)
    parser.add_argument("--grounding_path", type=str, default=None)
    parser.add_argument("--distributed_retrieval", type=int, default=16)
    parser.add_argument("--stride", type=int, default=5)
    args = parser.parse_args()
    return args

def load_predictions(path):
    global args
    paths = []
    if args.distributed_retrieval > 0:
        for i in range(args.distributed_retrieval):
            paths.append(path + f'/predictions_streaming_{str(i)}.txt')
            paths.append(path + f'/predictions_stream_{str(i)}.txt')
            paths.append(path + f'/predictions_negative_{str(i)}.txt')
    else:
        paths.append(path + f'/predictions.txt')
    logs = []
    for pp in paths:
        if os.path.isfile(pp):
            with open(pp) as f:
                for line in f:
                    try:
                        json_data = json.loads(line)
                        logs.append(json_data)
                    except Exception as e:
                        print(e, line)
    return logs

def iou(outputs, gt, num_frames_clip, num_frames_video, starts, indexes, single, hierarchy_zooms, grounding_windows):
    # return frames, ious
    frames = []
    clip_frames = {}
    for i, output in enumerate(outputs):
        matches = re.search(r"(\d+)", output)
        if matches:
            from_number = int(matches.group(1))
            from_number = from_number // hierarchy_zooms[i]
            if from_number< len(indexes[i]):
                from_number = indexes[i][from_number]
            from_number = starts[i] + from_number
            from_number = max(0, from_number)
            from_number = min(len(grounding_windows)-1,from_number)
            from_number = grounding_windows[from_number]
            to_number = from_number
            from_number = max(0, from_number-1)
            to_number = min(num_frames_video, to_number+1)
            clip_frames[i] = (int(from_number), int(to_number))
            frames.append((from_number, to_number))

    s, e = min(gt), max(gt)
    ious = []
    for frame in frames:
        f, t = frame#frame[0]/num_frames_video, frame[1]/num_frames_video
        intersection = max(0, min(t, e) - max(f, s))
        # union = max(t, e) - min(f, s)
        # iou = round(intersection / union, 2)
        ious.append(intersection)

    return clip_frames, [1] if sum(ious)>0 else [0]  # return frames, iou


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

def get_ground_truth_windows(start, end, duration):
    clip_len = 0.2
    start = start / clip_len
    end = end / clip_len
    slide_window_size = int(900 / 2)
    matched_subvideo_id_list = list(range(math.floor(start / slide_window_size),
                                     math.ceil(end / slide_window_size) + 1))
    duration = duration / clip_len
    duration = math.ceil(duration / slide_window_size) + 1
    return matched_subvideo_id_list, duration

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
    for id_item in js:
        id, item = id_item
        item['clip_id'] = id
        item['video_id'] = id
        item["timestamps"], item["duration"] = get_ground_truth_windows(item["timestamps"][0], item["timestamps"][1], item["movie_duration"])
    # features = torch.zeros((args.num_frames, 768), dtype=torch.float16)
    bin = len(js) // args.total_split
    items = js[args.split * bin:] if args.split == args.total_split - 1 else js[args.split * bin: (args.split + 1) * bin]
    batch = args.batch
    print('batch: ', batch)
    grounding_dict = {}
    if args.grounding_path is not None:
        grounding_logs = load_predictions(args.grounding_path)
        for gl in grounding_logs:
            grounding_dict[gl['query_id']] = gl
    for item in tqdm(items):
        # batch+=1
        # print('batch: ', batch)
        id, data = item
        i += 1
        if id in logs:
            continue
        try:
            movie = data['movie'] if 'movie' in data else data['clip_id']
            if args.vis_feat_storage == 'lmdb':  # because chapters data is enormous I use npy directly for now
                dump = appearance_visual_txn.get(movie.encode())
                with io.BytesIO(dump) as reader:
                    img_dump = np.load(reader, allow_pickle=True)
                    features = img_dump['features'] if 'features' in img_dump else img_dump['memory_global']
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
                continue

            ctx_l = len(features)
            assert ctx_l > 0, ctx_l
            clip_length = args.debug_window * args.feature_fps
            num_window = math.ceil(ctx_l / (clip_length//args.stride)) - 1
            windowidx = list(range(num_window))
            clip_feats = []
            times = []
            for i in windowidx:
                start = max(i * clip_length//args.stride, 0)
                end = min(i * clip_length//args.stride + clip_length, ctx_l-1)
                if end - start < clip_length:
                    start = end - clip_length
                times.append((start, end))
                sampled_indices = np.linspace(start, end, args.num_frames, dtype=np.int32)
                clip_feat = features[sampled_indices]
                clip_feats.append(clip_feat)
            if id in grounding_dict:
                grounding_windows = []
                grounding_windows_0 = [i for i, a in enumerate(grounding_dict[id]['answer']) if a != 'Not Present']
                for i in grounding_windows_0:
                    grounding_windows.extend(list(range(math.floor((i - 1) * (args.stride/2)), math.ceil((i - 1) * (args.stride/2) + (args.stride/2)))))
                grounding_windows = list(set(grounding_windows))
                # sort_index = [i for i, x in sorted(enumerate(grounding_dict[id]['info']['scores']), key=lambda x: x[1])]
                # grounding_windows = [grounding_windows[i] for i in sort_index]
                if batch>len(grounding_windows):
                    non_grounding_windows = [i for i in windowidx if i not in grounding_windows]
                    if len(non_grounding_windows)>0:
                        non_grounding_windows = non_grounding_windows[::int(len(non_grounding_windows)/(batch-len(grounding_windows)))][:batch-len(grounding_windows)]
                    grounding_windows = grounding_windows + non_grounding_windows
                    grounding_windows.sort()
                clip_feats = [clip_feats[i] for i in grounding_windows]
            else:
                grounding_windows = list(range(len(clip_feats)))

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
                query = 'During which video can we see {}?'#questions[args.mad_prompt]
                answers = []
                starts = []
                indexes = []
                mean_entropy = []
                max_entropy = []
                score_cos= []
                hierarchy_zooms = []
                for hierarchy_zoom in [4,2,1]:
                    batch = args.batch // hierarchy_zoom
                    for i in range(math.ceil(features.shape[0]/batch)):
                        start = i * batch
                        end = min(start + batch, features.shape[0])
                        if end - start < batch:
                            start = end - batch
                        starts.append(start)
                        feat = features[start:end][None]
                        if args.q_feat_dir is not None:# and i==0:
                            query_feats_temp = pad_sequences_1d(query_feats[None,].repeat(feat.shape[0],1,1), dtype=query_feats.dtype, device=query_feats.device, fixed_length=None)
                        idx = torch.randperm(feat.size(1))
                        feat = feat[:, idx]
                        indexes.append(idx)
                        if hierarchy_zoom>1:
                            feat = feat.repeat_interleave(hierarchy_zoom,1)
                        answer, model_output = inference(model, feat, query_feats_temp, "<video>\n" + query.format(sentence), tokenizer, return_list=True)
                        answers.extend(answer)
                        hierarchy_zooms.append(hierarchy_zoom)
                        entropy = get_entropy_statistics(torch.cat([a[:, None] for a in model_output['scores']], 1), 0,
                                                         model_output['scores'][0].shape[1])
                        max_entropy.extend([1/e[0].item() for e in entropy])
                        mean_entropy.extend([1/e[2].item() for e in entropy])
                        if args.single:
                            matches = re.search(r"(\d+)", answer[0])
                        else:
                            matches = re.search(r"(\d+) (to|and) (\d+)", answer[0])
                            if not matches:
                                matches = re.search(r"(\d+)", answer[0])

                        score = torch.tensor([0])
                        if matches:
                            from_number = int(matches.group(1))
                            from_number = from_number // hierarchy_zooms[i]
                            if from_number < len(indexes[i]):
                                from_number = indexes[i][from_number]
                            from_number = starts[i] + from_number
                            from_number = max(0, from_number)
                            from_number = min(len(grounding_windows) - 1, from_number)
                            from_number = grounding_windows[from_number]
                            to_number = from_number
                            from_number = max(0, from_number - 1)
                            to_number = min(to_number + 1, len(feat[0]) - 1)
                            score = []
                            for n in range(from_number,to_number):
                                feat_ = feat[:,n]
                                proposal_features = feat_ / feat_.norm(dim=1, keepdim=True)
                                proposal_features = _topk_pooling(query_cls_feats[None], proposal_features,min(proposal_features.shape[1], 3))[:, 0]
                                score.append(torch.einsum('bd,d->b', proposal_features, query_cls_feats))
                        score_cos.extend([a.item() for a in score])


                # if args.normalize:
                #     if len(max_entropy) > 0:
                #         m_s = max(max_entropy)
                #         max_entropy = [e / m_s for e in max_entropy]
                #     if len(mean_entropy) > 0:
                #         m_s = max(mean_entropy)
                #         mean_entropy = [e / m_s for e in mean_entropy]
                duration = data['movie_duration'] if 'movie_duration' in data else data['duration']
                # gt = (timestamps[0] / duration, timestamps[1] / duration)
                num_frames_video = args.batch
                frames, ious = iou(answers, timestamps, args.num_frames, num_frames_video, starts, indexes, args.single, hierarchy_zooms, grounding_windows)
                # scores=ious
                # scores = []
                # for k,v in frames.items():
                #     proposal_feat = features[k][v[0]:v[1]+1]
                #     proposal_features = proposal_feat / proposal_feat.norm(dim=0, keepdim=True)
                #     if args.topk_pool:
                #         proposal_features = _topk_pooling(query_cls_feats[None], proposal_features[None], min(proposal_features.shape[0], 3))[0]
                #         score = torch.einsum('bd,d->b', proposal_features, query_cls_feats)
                #     else:
                #         score = torch.einsum('bd,d->b', proposal_features, query_cls_feats).mean()
                #     scores.append(score.item())
                write_log(prediction_path, movie, 'grounding', id, answers, info={'gt':timestamps,
                                                                                  'frames': frames,
                                                                                  'iou': ious,
                                                                                  'score_cos': score_cos,
                                                                                  'mean_entropy': mean_entropy,
                                                                                  'max_entropy': max_entropy,
                                                                                  'hierarchy_zooms': hierarchy_zooms})
        except:
            if args.debug:
                raise
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