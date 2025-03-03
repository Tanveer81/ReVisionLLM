import argparse
import json
import random
import numpy as np
temporal_groundig_template = [
    'During which frames can we see {nlq_query} happening in the video?',
    'Between which frames is {nlq_query} visible in the video?',
    'At what point in the video can we observe {nlq_query} taking place?',
    'Between which two frames can we witness {nlq_query} occurring in the video?',
    'During which frames in the video can we observe {nlq_query} happening?',
    'At which time interval in the video can we see {nlq_query} occurring? ',
    'Between which frames can we find {nlq_query} taking place in the video?',
    'At what point in the video can we witness {nlq_query} happening?'
    'Between which two frames in the video can we observe {nlq_query} taking place?',
    'During which frames does {nlq_query} occur in the video?'
]

temporal_grounding_question_template = [
    "Pinpoint the time spans where the solution to the query can be found. The query is: {nlq_query}.",
    "Determine the intervals during which the answer to the query exists. The query is: {nlq_query}.",
    "Locate the time periods where the solution to the inquiry can be identified. The inquiry is: {nlq_query}.",
    "Find the time frames wherein the answer to the question lies. The question is: {nlq_query}.",
    "Discover the intervals during which the response to the inquiry is situated. The inquiry is: {nlq_query}.",
    "Uncover the time slots in which the solution to the query is present. The query is: {nlq_query}.",
    "Ascertain the time segments where the answer to the question resides. The question is: {nlq_query}.",
    "Identify the intervals during which the solution to the query can be determined. The query is: {nlq_query}.",
    "Pin down the time ranges where the response to the inquiry can be located. The inquiry is: {nlq_query}.",
    "Determine the time slots within which the answer to the query can be identified. The query is: {nlq_query}."
]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chapters_data_path", type=str,
                        default='/nfs/data3/user/revisionllm/activitynet/val_2.json')
    parser.add_argument("--activitynet_data_path", type=str, default="/nfs/data3/user/revisionllm/stage2.json")
    parser.add_argument("--chapters_out_path", type=str, default="/nfs/data3/user/revisionllm/activitynet_val_2.json")
    args = parser.parse_args()
    return args


def convert_ego_to_vtimellm(ego_data):
    #{"1050": {"movie": "3003_40_YEAR_OLD_VIRGIN",
             # "sentence": "He enters a store called Smart Tech carrying his front bicycle tire and bag.",
             # "timestamps": [162.547, 175.809], "ext_timestamps": [162.547, 175.809],
             # "movie_duration": 7947.6
    ego_to_activity = {}
    time=[]
    for key, value in mad_data.items():
        for sentence_id, (timestamps, sentence) in enumerate(zip(value['timestamps'], value['sentences'])):
            second_dict = {}
            second_dict['sentence'] = sentence
            second_dict['timestamps'] = timestamps
            second_dict['movie_duration'] = value["duration"]
            #second_dict['query_id'] = first_dict['qid']
            second_dict['movie'] = key
            second_dict['query_id'] = key + '_' + str(sentence_id)
            second_dict['split'] = 'val'
            ego_to_activity[key+'_'+str(sentence_id)] = second_dict
            time.append([(timestamps[1] - timestamps[0]), value["duration"]])

    return ego_to_activity


if __name__ == "__main__":
    args = parse_args()
    mad_data = json.load(open(args.chapters_data_path))
    chapters_to_activity = convert_ego_to_vtimellm(mad_data)
    with open(args.chapters_out_path, 'w') as f:
        json.dump(chapters_to_activity, f)
