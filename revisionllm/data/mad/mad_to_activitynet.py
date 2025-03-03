import argparse
import json
import random

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
    parser.add_argument("--mad_data_path", type=str,
                        default='/data/mad/MAD_train.json')
    parser.add_argument("--mad_out_path", type=str,
                        default="/data/mad/mad_stage2_fixed_prompt_neg.json")
    parser.add_argument("--neg", type=bool, default=False)
    args = parser.parse_args()
    return args


def convert_mad_to_vtimellm(mad_data):
    mad_to_activity = []
    for key, value in mad_data.items():
        second_dict = {}
        second_dict['query_id'] = key
        second_dict['id'] = value['movie']
        second_dict['conversations'] = []
        token = {}
        conversation = {}
        conversation['from'] = 'human'
        sentence = value['sentence'].strip().lower()
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

        token[f'<s0>'] = round(value["timestamps"][0], 1)
        token[f'<e0>'] = round(value["timestamps"][1], 1)
        #'split': [round(first_dict['clips'][0]['video_start_sec'], 1), round(first_dict['clips'][0]['video_end_sec'], 1)],
        second_dict['meta'] = {'duration': value["movie_duration"],
                               'token': token}
        second_dict['source'] = 'mad'
        mad_to_activity.append(second_dict)
        if args.neg:
            second_dict['neg'] = 'yes'
            mad_to_activity.append(second_dict)
    return mad_to_activity


if __name__ == "__main__":
    args = parse_args()
    # ad = json.load(open(args.activitynet_data_path))
    mad_data = json.load(open(args.mad_data_path))
    mad_to_activity = convert_mad_to_vtimellm(mad_data)
    with open(args.mad_out_path, 'w') as f:
        json.dump(mad_to_activity, f)
