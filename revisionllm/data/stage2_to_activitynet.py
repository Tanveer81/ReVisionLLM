import argparse
import json
import re

dense_caption_template = [
    'Could you please detail the events that took place during different time segments in the video?',
    'I’m curious about what happened at different points in the video. Could you please describe the events?',
    'Could you provide a summary of the incidents that occurred at various timestamps in the video?',
    'I’d like to know what events transpired during specific time intervals in the video.',
    'Could you please elaborate?',
    'Can you give me a breakdown of the occurrences at different time stamps in the video?',
    'I’m interested in understanding the events that unfolded at different points in the video. Could you please specify?',
    'Could you outline the incidents that happened during different time periods in the video?',
    'I’m trying to grasp the sequence of events in the video. Could you please outline what happened at different times?',
    'Can you go through the video and describe what took place at different time intervals?',
    'I’d appreciate it if you could provide a detailed account of the events that occurred at different timestamps in the video.'
]
event_caption_template = [
 'Can you describe what occurred from <s(.+?)> to <e(.+?) in the video?',
 'Could you tell me what happened from <s(.+?)> to <e(.+?) in the video?',
 'What transpired from <s(.+?)> to <e(.+?) in the video?',
 'Describe what took place from <s(.+?)> to <e(.+?) in the video.',
 'Tell me about the events from <s(.+?)> to <e(.+?) in the video.',
 'What was going on from <s(.+?)> to <e(.+?) in the video?',
 'Please recount what occurred from <s(.+?)> to <e(.+?) in the video.',
 'Explain what happened from <s(.+?)> to <e(.+?) in the video.',
 'Provide details about the events from <s(.+?)> to <e(.+?) in the video.',
 'Share what transpired from <s(.+?)> to <e(.+?) in the video.'
]
temporal_groundig_template = [
    'During which frames can we see (.+?) happening in the video?',
    'Between which frames is (.+?) visible in the video?',
    'At what point in the video can we observe (.+?) taking place?',
    'Between which two frames can we witness (.+?) occurring in the video?',
    'During which frames in the video can we observe (.+?) happening?',
    'At which time interval in the video can we see (.+?) occurring?',
    'Between which frames can we find (.+?) taking place in the video?',
    'At what point in the video can we witness (.+?) happening?'
    'Between which two frames in the video can we observe (.+?) taking place?',
    'During which frames does (.+?) occur in the video?'
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
                        default="/nfs/data3/user/revisionllm/stage2.json")
    parser.add_argument("--activitynet_data_path", type=str, default="/nfs/data3/user/revisionllm/stage2.json")
    parser.add_argument("--mad_out_path", type=str,
                        default="/nfs/data3/user/revisionllm/stage2_train.json")
    parser.add_argument("--neg", type=bool, default=False)
    args = parser.parse_args()
    return args


def convert_mad_to_vtimellm(mad_data):
    mad_to_activity = []
    time=[]
    for kk, value in enumerate(mad_data):
        if kk==1155:
            print()
        id2 = 0
        for sentence_id, sentence in enumerate(value['conversations']):
            sentence = sentence['value'].strip().lower()
            if re.search('^from <s(.+?)> to <e(.+?)>$', sentence):
                continue
            sentence = re.sub('<video>\n', '', sentence)
            for template in event_caption_template:
                m = re.search(template.strip().lower(), sentence)
                if m:
                    break
            if m:
                continue
            if re.search('from <s(.+?)> to <e(.+?)>', sentence):
                sentence = re.sub('from <s(.+?)> to <e(.+?)>, ', '', sentence)
                sentence = re.sub(', from <s(.+?)> to <e(.+?)>', '', sentence)
                sentence = re.sub('from <s(.+?)> to <e(.+?)>', '', sentence)
                sentences = sentence.split('.')
                id2=-1
            else:
                for template in temporal_groundig_template:
                    m = re.search(template.strip().lower(), sentence)
                    if m:
                        sentences = [m.group(1)]
                        break
                if not m:
                    continue
            for sentence in sentences:
                if len(sentence)==0 or sentence == ' ':
                    continue
                second_dict = {}
                second_dict['query_id'] = value['id'] + '_' + str(sentence_id+id2)
                second_dict['id'] = value['id']
                second_dict['conversations'] = []
                token = {}
                conversation = {}
                conversation['from'] = 'human'
                sentence = sentence.strip().lower()
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
                timestamps = (list(value['meta']['token'].values())[((sentence_id+id2)//2)*2],
                              list(value['meta']['token'].values())[((sentence_id+id2)//2)*2+1])
                token[f'<s0>'] = round(timestamps[0], 1)
                token[f'<e0>'] = round(timestamps[1], 1)
                #'split': [round(first_dict['clips'][0]['video_start_sec'], 1), round(first_dict['clips'][0]['video_end_sec'], 1)],
                second_dict['meta'] = {'duration': value['meta']["duration"],
                                       'token': token,
                                       'split': value['meta']['split']}
                second_dict['source'] = 'stage2'
                mad_to_activity.append(second_dict)
                if args.neg:
                    second_dict['neg'] = 'yes'
                    mad_to_activity.append(second_dict)
                time.append([(timestamps[1] - timestamps[0]), value['meta']["duration"]])
                id2+=1
    return mad_to_activity


if __name__ == "__main__":
    args = parse_args()
    # ad = json.load(open(args.activitynet_data_path))
    mad_data = json.load(open(args.mad_data_path))
    mad_to_activity = convert_mad_to_vtimellm(mad_data)
    with open(args.mad_out_path, 'w') as f:
        json.dump(mad_to_activity, f)
