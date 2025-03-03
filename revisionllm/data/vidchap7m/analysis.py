import argparse
import collections
import json
import os
import random

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--chapters_data_path", type=str,
                        default="/nfs/data3/user/AllChapters/chapters_vmr_test.jsonl")
    parser.add_argument("--activitynet_data_path", type=str, default="/nfs/data3/user/revisionllm/stage2.json")
    parser.add_argument("--chapters_out_path", type=str, default="/nfs/data3/user/revisionllm/chapters_test.json")
    args = parser.parse_args()
    return args


def stat(ego_data):
    ego_to_activity = {}
    path = f'/nfs/data3/user/AllChapters/test.json'
    with open(path) as f:
        test_vodeos = json.load(f)
    timestamps = []
    movie_duration = []
    for first_dict in ego_data:
        timestamps.append(first_dict['relevant_windows'][0][1] - first_dict['relevant_windows'][0][0])
        movie_duration.append(first_dict['duration'])
    return ego_to_activity


def test_stat():
    if 'jsonl' in args.chapters_data_path:
        with open(args.chapters_data_path) as f:
            chapters_data = [json.loads(line) for line in f]
    else:
        chapters_data = json.load(open(args.chapters_data_path))['videos']
    chapters_to_activity = stat(chapters_data)


def video_category():
    '''
    info_not_found:  12
    category_not_found:  3
    defaultdict(<class 'int'>, {'Howto & Style': 11258,
                                'Entertainment': 7802,
                                'People & Blogs': 9199,
                                'Sports': 3214,
                                'Science & Technology': 6201,
                                'Travel & Events': 2018,
                                'Music': 3557,
                                'Education': 9853,
                                'Autos & Vehicles': 2993,
                                'News & Politics': 1352,
                                'Gaming': 4042,
                                'Comedy': 375,
                                'Film & Animation': 1710,
                                'Pets & Animals': 513,
                                'Nonprofits & Activism': 371})
    '''
    info_not_found = 0
    category_not_found = 0
    category = collections.defaultdict(int)
    train_path = "/nfs/data3/user/AllChapters/chapters_vmr_train.jsonl"
    with open(train_path) as f:
        train_data = [json.loads(line) for line in f]

    for first_dict in tqdm(train_data):
        try:
            path = f"/nfs/data3/user/revisionllm/chapters_clipl14_features/{first_dict['vid']}.npy"
            if not os.path.isfile(path):
                continue
            video_info_path = f"/nfs/data3/user/AllChapters/train_video_info/{first_dict['vid']}.info.json"
            if not os.path.isfile(video_info_path):
                info_not_found +=1
                continue
            info = json.load(open(video_info_path))
            if 'categories' not in info:
                category_not_found +=1
            else:
                category[json.load(open(video_info_path))['categories'][0]] += 1
        except Exception as e:
            print(e)
            print(category)

    print('info_not_found: ', info_not_found)
    print('category_not_found: ', category_not_found)
    print(category)
    print()


if __name__ == "__main__":
    args = parse_args()
    #video_category()
    # ad = json.load(open(args.activitynet_data_path))
    test_stat()

