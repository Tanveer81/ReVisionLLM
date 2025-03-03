import argparse
import json
import os
import random

from tqdm import tqdm

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
                        default="/data/chapters/chapters_vmr_train.jsonl")
    parser.add_argument("--activitynet_data_path", type=str, default="/data/stage2.json")
    parser.add_argument("--chapters_out_path", type=str, default="/data/chapters/chapters_train_partial_3.json")
    args = parser.parse_args()
    return args


def convert_ego_to_vtimellm(ego_data):
    ego_to_activity = []
    for first_dict in tqdm(ego_data):
        i = 0
        for query, window in zip(first_dict['query'], first_dict['relevant_windows']) :
            second_dict = {}
            path = f"/data/chapters/chapters_clipl14_features/{first_dict['vid']}.npy"
            if not os.path.isfile(path):
                continue
            second_dict['id'] = first_dict['vid'] #+ '_' + str(first_dict['qid'])
            second_dict['query_id'] = first_dict['vid']+'_'+ str(i)#first_dict['qid']
            i+=1
            second_dict['conversations'] = []
            token = {}
            conversation = {}
            conversation['from'] = 'human'
            sentence = query.strip().lower()
            if sentence.endswith("."):
                sentence = sentence[:-1]
            # prompt = temporal_groundig_template[random.randint(0, 8)].format(nlq_query=sentence)#.rstrip("?,.'"))
            prompt = 'During which frames can we see {}?'.format(sentence)
            conversation['value'] = f'<video>\n{prompt}'
            second_dict['conversations'].append(conversation)

            conversation = {}
            conversation['from'] = 'gpt'
            conversation['value'] = f'From <s0> to <e0>.'
            second_dict['conversations'].append(conversation)

            token[f'<s0>'] = round(window[0][0], 1)
            token[f'<e0>'] = round(window[0][1], 1)

            second_dict['meta'] = {'duration': first_dict['duration'], 'token': token}
            second_dict['source'] = 'vidchapters7m'

            ego_to_activity.append(second_dict)
    return ego_to_activity


if __name__ == "__main__":
    args = parse_args()
    # ad = json.load(open(args.activitynet_data_path))
    if 'jsonl' in args.chapters_data_path:
        with open(args.chapters_data_path) as f:
            chapters_data = [json.loads(line) for line in f]
    else:
        chapters_data = json.load(open(args.chapters_data_path))['videos']
    chapters_to_activity = convert_ego_to_vtimellm(chapters_data)
    with open(args.chapters_out_path, 'w') as f:
        json.dump(chapters_to_activity, f)
