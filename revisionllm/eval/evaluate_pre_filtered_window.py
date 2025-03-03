"""
Script to evaluate performance of coarse-grained window selection module of CONE
"""

import math
import torch
import numpy as np
import tqdm
import terminaltables
from utils.basic_utils import load_jsonl
import utils.misc as util

def display_window_results(results, topK, title=None):
    display_data = [
        [f"Rank@{ii}" for ii in topK]
    ]
    results *= 100
    # print(results)
    display_data.append(
        [
            f"{results[ii]:.02f}"
            for ii in range(len(topK))
        ]
    )
    table = terminaltables.AsciiTable(display_data, title)
    for ii in range(len(topK)):
        table.justify_columns[ii] = "center"
    return table.table, display_data


def windows_selection(
        query_id2windowidx, ground_truth, topK, opt=None, match_number=True,
):
    pred_qids = set(query_id2windowidx.keys())
    gt_qids = set([e["query_id"] for e in ground_truth])

    if match_number:
        assert pred_qids == gt_qids, \
            f"qids in ground_truth and submission must match. " \
            f"use `match_number=False` if you wish to disable this check"
    else:  # only leave the items that exists in both submission and ground_truth
        shared_qids = pred_qids.intersection(gt_qids)
        query_id2windowidx = {k: v for k, v in query_id2windowidx.items() if k in shared_qids}
        ground_truth = [e for e in ground_truth if e["query_id"] in shared_qids]

    truth_dict = {}
    for _meta in ground_truth:
        clip_len = opt.clip_length
        start = _meta["timestamps"][0] / clip_len
        end = _meta["timestamps"][1] / clip_len
        slide_window_size = int(opt.max_v_l / 2)
        matched_subvideo_id_list = range(math.floor(start / slide_window_size),
                                         math.ceil(end / slide_window_size) + 1)
        truth_dict[_meta['query_id']] = torch.Tensor(matched_subvideo_id_list)

    recall_metrics = topK#torch.tensor(topK)
    max_recall = recall_metrics.max()
    num_recall_metrics = len(recall_metrics)
    recall_x = torch.zeros(num_recall_metrics)

    window_number_list = []
    for query_id, window_list in tqdm.tqdm(query_id2windowidx.items()):
        true_window = torch.tensor(truth_dict[query_id])
        window_number_list.append(len(window_list))
        bools = torch.tensor([idx in true_window for idx in window_list[:max_recall]])
        for i, r in enumerate(recall_metrics):
            recall_x[i] += bools[:r].any(dim=0)
    window_number_array = np.array(window_number_list)
    if util.is_main_process():
        print("avg window number: ", np.mean(window_number_array))
        print("median window number: ", np.median(window_number_array))

    recall_x /= len(query_id2windowidx)
    return recall_x


def get_ground_truth_windows(opt):
    ground_truth = load_jsonl(opt.eval_path)
    truth_dict = {}
    for _meta in ground_truth:
        clip_len = opt.clip_length
        start = _meta["timestamps"][0] / clip_len
        end = _meta["timestamps"][1] / clip_len
        slide_window_size = int(opt.max_v_l / 2)
        matched_subvideo_id_list = list(range(math.floor(start / slide_window_size),
                                         math.ceil(end / slide_window_size) + 1))
        truth_dict[_meta['query_id']] = matched_subvideo_id_list
    return truth_dict