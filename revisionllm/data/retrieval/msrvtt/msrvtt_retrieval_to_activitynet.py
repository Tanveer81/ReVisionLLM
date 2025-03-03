import argparse
import json
import math
import random
import pandas as pd
from csv import DictReader


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mad_data_path", type=str,
                        default='/nfs/data8/user/msrvtt_data/MSRVTT_JSFUSION_test.csv')
    parser.add_argument("--activitynet_data_path", type=str, default="/nfs/data3/user/revisionllm/stage2.json")
    parser.add_argument("--mad_out_path", type=str,
                        default="/nfs/data8/user/msrvtt_data/msrvtt_train_hierarchy.json")#"/home/atuin/v100dd/v100dd11/mad/annotations/MAD-v1/mad_retrieval.json")
    parser.add_argument("--neg", type=bool, default=False)
    args = parser.parse_args()
    return args


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

def convert_mad_to_vtimellm(mad_data, msrvtt_train_9k):
    mad_to_activity = []
    for value in mad_data['sentences']:
        if value['video_id'] not in msrvtt_train_9k:
            continue
        second_dict = {}
        second_dict['query_id'] = value['sen_id']
        second_dict['id'] = value['video_id']
        second_dict['conversations'] = []
        token = {}
        conversation = {}
        conversation['from'] = 'human'
        sentence = value['caption'].strip().lower()
        if sentence.endswith("."):
            sentence = sentence[:-1]
        # prompt = temporal_groundig_template[random.randint(0, 8)].format(nlq_query=sentence)#.rstrip("?,.'"))
        # prompt = 'Does {} happen in the video? Write your answer either yes or no.'.format(sentence)
        prompt = 'During which video can we see {}?'.format(sentence)
        conversation['value'] = f'<video>\n{prompt}'
        second_dict['conversations'].append(conversation)

        conversation = {}
        conversation['from'] = 'gpt'
        conversation['value'] = 'yes'#f'From <s0> to <e0>.'
        second_dict['conversations'].append(conversation)

        # clips, duration = get_ground_truth_windows(value["timestamps"][0], value["timestamps"][1], value["duration"])
        # token[f'<s0>'] = ''#clips[0]#round(value["timestamps"][0], 1)
        # token[f'<e0>'] = ''#clips[1]#round(value["timestamps"][1], 1)
        #'split': [round(first_dict['clips'][0]['video_start_sec'], 1), round(first_dict['clips'][0]['video_end_sec'], 1)],
        # second_dict['meta'] = {'duration': '', #test min 56 max 131 train min 33 max 136
        #                        'token': token}
        second_dict['source'] = 'msrvtt'
        mad_to_activity.append(second_dict)
        if args.neg:
            second_dict['neg'] = 'yes'
            mad_to_activity.append(second_dict)
    return mad_to_activity


if __name__ == "__main__":
    args = parse_args()
    # ad = json.load(open(args.activitynet_data_path))
    msrvtt_data = json.load(open('/nfs/data8/user/msrvtt_data/MSRVTT_data.json'))

    # open file in read mode
    with open('/nfs/data8/user/msrvtt_data/MSRVTT_train.9k.csv', 'r') as f:
        dict_reader = DictReader(f)
        msrvtt_train_9k = list(dict_reader)
        msrvtt_train_9k = [mt['video_id'] for mt in msrvtt_train_9k]

    mad_to_activity = convert_mad_to_vtimellm(msrvtt_data, msrvtt_train_9k)
    with open(args.mad_out_path, 'w') as f:
        json.dump(mad_to_activity, f)
