import argparse
import json

if __name__ == '__main__':
    for split in ['test', 'train', 'val']:
        print(split)
        path = f'/nfs/data3/user/AllChapters/{split}.json'
        with open(path) as f:
            d = json.load(f)

    total_split=10
    bin = len(d) // total_split
    for i in range(total_split):
        items = d[i*bin:] if i == total_split-1 else d[i*bin:(i+1)*bin]
        for video in items:
            with open(f'/nfs/data3/user/AllChapters/train_split/{split}_{i}.txt', 'a') as f:
                f.write(f'{video}\n')