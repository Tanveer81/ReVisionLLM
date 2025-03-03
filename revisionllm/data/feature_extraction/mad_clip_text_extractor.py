"""
Utility: extract clip-based textual eot feature and textual token feature into a single lmdb file for MAD dataset
"""
import torch
from basic_utils import load_jsonl
import numpy as np
import tqdm
import lmdb
import msgpack
import io
from clip_extractor import ClipFeatureExtractor
from torch.utils.data import DataLoader, Dataset
import argparse
import clip
from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer

_tokenizer = _Tokenizer()

def dumps_msgpack(dump):
    return msgpack.dumps(dump, use_bin_type=True)


def dumps_npz(dump, compress=False):
    with io.BytesIO() as writer:
        if compress:
            np.savez_compressed(writer, **dump, allow_pickle=True)
        else:
            np.savez(writer, **dump, allow_pickle=True)
        return writer.getvalue()


class SingleSentenceDataset(Dataset):
    def __init__(self, input_datalist, block_size=512, debug=False):
        self.max_length = block_size
        self.debug = debug
        self.examples = input_datalist

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        self.examples[index]['query'] = self.process_query(self.examples[index]['query'])
        return self.examples[index]

    def process_query(self,question):
        """Process the query to make it canonical."""
        return question.strip(".").strip(" ").strip("?") + "."


def pad_collate(data):
    batch = {}
    for k in data[0].keys():
        batch[k] = [d[k] for d in data]
    return batch


def extract_mad_text_feature(args):
    #format = "./data/mad_data/%s.jsonl"
    format = "/data/mad/%s.jsonl"
    split_list = ['train', 'test', 'val']
    total_data = []
    for split in split_list:
        filename = format % split
        data = load_jsonl(filename)
        total_data.extend(data)
    print(len(total_data))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Build models...")
    clip_model_name_or_path = "ViT-L/14"
    feature_extractor = ClipFeatureExtractor(
        framerate=30, size=224, centercrop=True,
        model_name_or_path=clip_model_name_or_path, device=device
    )

    dataset = SingleSentenceDataset(input_datalist=total_data)

    eval_dataloader = DataLoader(dataset, batch_size=60, collate_fn=pad_collate)

    feature_save_path = args.feature_output_path
    feature_save_path = '/data/mad/clip_L14_text_features'
    text_output_env = lmdb.open(feature_save_path, map_size=1099511627776)

    for batch in tqdm.tqdm(eval_dataloader, desc="Evaluating", total=len(eval_dataloader)):
        query_id_list = batch["query_id"]
        query_list = batch["query"]
        token_features, text_eot_features = feature_extractor.encode_text(query_list)

        # for i in range(len(query_list)):
        #     if i == 0:
        #         token_feat = np.array(token_features[i].detach().cpu()).astype(np.float32)
        #         eot_feat = np.array(text_eot_features[i].detach().cpu()).astype(np.float32)
        #         print("query: ", query_list[i])
        #         #print("query tokenize 0: ", _tokenizer.bpe(query_list[i]))
        #         encode_text = _tokenizer.encode(query_list[i])
        #         print("query tokenize 1: ", encode_text)
        #         print("query tokenize idx: ", clip.tokenize(query_list[i]))
        #         print("decoder query: ", _tokenizer.decode(encode_text))
        #         print("token_feat: ", token_feat.shape)
        #         print("text_eot_features: ", eot_feat.shape)

        with text_output_env.begin(write=True) as text_output_txn:
            for i in range(len(query_list)):
                q_feat = np.array(text_eot_features[i].detach().cpu()).astype(np.float32)
                token_feat = np.array(token_features[i].detach().cpu()).astype(np.float32)
                features_dict = {"cls_features": q_feat, "token_features": token_feat}
                feature_dump = dumps_npz(features_dict, compress=True)
                text_output_txn.put(key=query_id_list[i].encode(), value=feature_dump)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature_output_path", help="Path to train split"
    )  # "/s1_md0/leiji/v-zhijian/MAD_data/CLIP_text_features"
    args = parser.parse_args()
    extract_mad_text_feature(args)
