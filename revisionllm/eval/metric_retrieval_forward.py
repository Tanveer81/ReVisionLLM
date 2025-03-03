import collections
import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# from dvc_eval import eval_dvc, eval_soda

import json
import argparse
import re
import difflib

def print_metrics(metrics):
    for k, v in metrics.items():
        print(f"{k}: {v:.2f}")

def grounding_metrics(all_logs):
    ious = [x['info']['iou'] for x in all_logs if x['task'] == 'grounding' and x['info']['iou']!=-1]
    fn = [x['info']['fn'] for x in logs if x['task'] == 'grounding' and 'fn' in x['info']]
    fp = [x['info']['fp'] for x in logs if x['task'] == 'grounding' and 'fp' in x['info']]
    l = len(ious)
    lf = len(fp) // 2
    print(f"Found {l} logs")
    if l == 0: return
    metrics = {
        "mIoU": sum(ious) / l * 100,
        "fn": sum(fn) / lf * 100,
        "fp": sum(fp) / lf * 100
    }
    for m in [0.1, 0.3, 0.5, 0.7]:
        metrics[f"R1@{m}"] = sum(iou >= m for iou in ious) / l * 100
    return metrics

def grounding_metrics_stream(all_logs):
    ious = []
    for log in all_logs:
        try:
            sorted_idx = sorted(range(len(log['info']['scores'])), key=lambda k: log['info']['scores'][k], reverse=True)
            iou = [log['info']['iou'][i] for i in sorted_idx]
            ious.append(np.array(iou))
        except:
            ious.append(np.array([log['info']['iou']]))
    l = len(ious)
    print(f"Found {l} logs")
    if l == 0: return
    metrics = collections.defaultdict(int)
    metrics["mIoU"] = sum([u[0] for u in ious if len(u)>=1]) / l * 100

    for m in [0.1, 0.3, 0.5, 0.7, 0.9]:
        for iou in ious:
            bools = iou > m
            for r in [1, 5, 10, 50]:
                metrics[f"R{r}@{m}"] += bools[:r].any() / l * 100

    return metrics


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--grounding_path", type=str, default='checkpoints/stage1_dense')
    parser.add_argument("--retrieval_path", type=str, default='checkpoints/stage2_long_100')
    parser.add_argument("--retrieval_path2", type=str, default='checkpoints/stage2_long_33')
    parser.add_argument("--task", type=str, default='grounding', choices=['all', 'grounding', 'captioning'])
    parser.add_argument("--data_path", type=str, default='revisionllm/eval/data_example.json')
    parser.add_argument("--stream", type=bool, default=True)
    parser.add_argument("--distributed_grounding", type=int, default=16)
    parser.add_argument("--distributed_retrieval", type=int, default=16)
    parser.add_argument("--single", type=bool, default=True)

    args = parser.parse_args()

    grounding_logs = load_predictions(args.grounding_path)
    retrieval_logs = load_predictions(args.retrieval_path)
    retrieval_dict = {}
    for rl in retrieval_logs:
        retrieval_dict[rl['query_id']] = rl
    if args.retrieval_path2 is not None:
        retrieval_logs2 = load_predictions(args.retrieval_path2)
        retrieval_dict2 = {}
        for rl in retrieval_logs2:
            retrieval_dict2[rl['query_id']] = rl

    for buffer in [0]:
        print('buffer: ', buffer)
        grunding_dict = []
        total = []
        selected = []
        for gl in grounding_logs:
            # total.append(len(gl['answer']))
            if gl['query_id'] in retrieval_dict:
                rl = retrieval_dict[gl['query_id']]
                if args.retrieval_path2 is not None:
                    rl2 = retrieval_dict2[gl['query_id']]
                frames = []
                clip_frames = {}
                # buffer=1
                gl_idx = [i for i, a in enumerate(gl['answer']) if a != 'Not Present' and a != 'From 249 to 249.']
                if args.single:
                    for i, output in enumerate(rl['answer']):
                        for i, output in enumerate(list(rl['info']['frames'].values())):
                            frames.extend(list(range(max(0, int(.4 * output[0]) - buffer),
                                                     min(int(.4 * output[1]) + buffer, len(gl['answer']) - 1))))
                    present_idx1 = [i for i in gl_idx if i in frames]

                    if args.retrieval_path2 is not None:
                        if 'frames' in rl2['info']:
                            for i, output in enumerate(list(rl2['info']['frames'].values())):
                                frames.extend(list(range(max(0, int(.4 * output[0]) - buffer),
                                                         min(int(.4 * output[1]) + buffer, len(gl['answer']) - 1))))
                else:
                    matches = re.search(r"(\d+) (to|and) (\d+)", output)
                    if matches:
                        from_number = int(matches.group(1))
                        to_number = int(matches.group(3))
                        frames.extend(list(range(max(0, from_number-buffer), min(to_number+buffer, len(gl['answer'])-1))))
                frames = list(set(frames))
                total.append(len(gl['answer']))
                present_idx = [i for i in gl_idx if i in frames]
                if len(present_idx1)>0 and buffer!=-1:
                    answer = [gl['answer'][i] for i in present_idx]
                    iou = [gl['info']['iou'][gl_idx.index(i)] for i in present_idx]
                    # if len(rl['info']['score_cos']) > 0:
                    #     m_s = max(rl['info']['score_cos'])
                    #     rl['info']['score_cos'] = [e / m_s for e in rl['info']['score_cos']]
                    if len(gl['info']['scores']) > 0:
                        min_s = min(gl['info']['scores'])
                        max_s = max(gl['info']['scores'])
                        if min_s != max_s:
                            gl['info']['scores'] = [(gl['info']['scores'][i] - min_s) / (max_s - min_s) for i in
                                                    range(len(gl['info']['scores']))]

                    if len(rl['info']['mean_entropy']) > 0:
                        min_s = min(rl['info']['mean_entropy'])
                        max_s = max(rl['info']['mean_entropy'])
                        rl['info']['mean_entropy'] = [(rl['info']['mean_entropy'][i] - min_s) / (max_s - min_s) for i in
                                                      range(len(rl['info']['mean_entropy']))]

                    # scores = [gl['info']['scores'][gl_idx.index(i)] + rl['info']['mean_entropy'][frames.index(i)//(2*buffer+1)] for i in present_idx]
                    # scores = [gl['info']['scores'][gl_idx.index(i)] * (rl['info']['score_cos'][frames.index(i)//(2*buffer+1)]/rl['info']['mean_entropy'][frames.index(i)//(2*buffer+1)]) for i in present_idx]
                    scores = [gl['info']['scores'][gl_idx.index(i)] for i in present_idx]
                    # scores = [gl['info']['scores'][gl_idx.index(i)] * rl['info']['mean_entropy'][frames.index(i)//(2*buffer+1)] for i in present_idx]

                    # gl['answer'] = answer
                    # gl['info']['iou'] = iou
                    # gl['info']['scores'] = scores
                    for a in answer:
                        if a != 'Not Present':
                            gl['answer'] = answer
                            gl['info']['iou'] = iou
                            gl['info']['scores'] = scores
                            break
                    # if len(gl['info']['scores']) > 0:
                    #     m_s = max(gl['info']['scores'])
                    #     m_s = 1 if m_s==0 else m_s
                    #     gl['info']['scores'] = [e / m_s for e in gl['info']['scores']]
                #     selected.append(len(frames))
                # else:
                selected.append(len(gl['answer']))
                grunding_dict.append(gl)

        print(args.grounding_path)
        print(sum(selected)/sum(total))
        if args.task in ['captioning', 'all']:
            print("====================== Captioning =====================")
            print_metrics(captioning_metrics(logs, args.data_path))
        if args.task in ['grounding', 'all']:
            print("====================== Grounding ======================")
            if args.stream:
                metrics = grounding_metrics_stream(grunding_dict)
                # metrics = grounding_metrics_stream(grounding_logs)
            else:
                metrics = grounding_metrics(grunding_dict)
            print_metrics(metrics)
            with open(args.grounding_path + '/result_retrieval.txt', 'w+') as f:
                json.dump(metrics, f)
