import collections
import io
import math
import random
import copy
import json
import re
from csv import DictReader

import numpy
import torch
import transformers
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import os.path

from revisionllm.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IGNORE_TOKEN
from revisionllm import conversation as conversation_lib
from revisionllm.mm_utils import tokenizer_image_token
import lmdb
import copy
from revisionllm.model.adapter.tensor_utils import pad_sequences_1d


@dataclass
class DataArguments:
    data_path: str = field(default=None, metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    feat_folder: Optional[str] = field(default=None)
    q_feat_dir: Optional[str] = field(default=None)
    num_frames: int = 100
    hierarchy_num_videos: int = 100
    hierarchy_zoom: bool = field(default=False)
    fix_hierarchy_zoom: int = 0
    hierarchy_neg: bool = field(default=False)
    dataset: str = 'mad'
    debug_window: int = 0
    max_q_l: int = 25
    neg_window: bool = field(default=False)
    neg_samples: float = field(default=1.0)
    neg_factor: int = field(default=1)
    retrieval_only: bool = field(default=False)
    stream: bool = field(default=False)
    feature_fps: float = 5.0
    debug_my_dataset: bool = False
    vis_feat_storage: str = 'lmdb'
    keep_longer_gt: bool = False
    ignore_temporal: bool = field(default=False)
    t2v: str = field(default=None, metadata={"help":'/nfs/data8/user/msrvtt_data/MSRVTT_train.9k.csv'})
    sparse_dataset: bool = False
    sparse_length: int = 0
    long_baseline: bool = False

def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_glm(
    sources,
    tokenizer: transformers.PreTrainedTokenizer
) -> Dict:
    
    input_ids = []
    targets = []

    for source in sources:
        tokens, loss_masks = [tokenizer.get_command("[gMASK]"), tokenizer.get_command("sop")], [0, 0]
        def _update(_tokens: List[int], value: int = 1):
            value = int(value)
            tokens.extend(_tokens)
            loss_masks.extend([value] * len(_tokens))
        
        for conv in source:
            if conv["from"] == 'human':
                role_token = tokenizer.get_command("<|user|>")
                loss = False
            else:
                role_token = tokenizer.get_command("<|assistant|>")
                loss = True
                
            token_id = [role_token] + tokenizer_image_token(conv['value'], tokenizer)[2:]
            _update(token_id, loss)
        _update([tokenizer.eos_token_id], False)

        loss_masks = [False] + loss_masks[:-1]
        labels = [(t if m else IGNORE_INDEX) for t, m in zip(tokens, loss_masks)]

        input_ids.append(tokens)
        targets.append(labels)

        # print("Sanity Check >>>>>>>>>>>>>")
        # for t, m in zip(tokens, labels):
        #     decoded =  tokenizer.tokenizer.index_special_tokens[t] \
        #         if t in tokenizer.tokenizer.index_special_tokens \
        #         else tokenizer.decode([t])
        #     print("%20s: %6d -> %6d" % (repr(decoded), t, m))
        # print("<<<<<<<<<<<<< Sanity Check")

    return dict(
        input_ids=torch.tensor(input_ids),
        labels=torch.tensor(targets),
    )

def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    ignore_temporal: bool = False,
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
    # print(conversations)
    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
    if ignore_temporal:
        numeric_tokens = [tokenizer(str(i),return_tensors="pt",padding="longest",max_length=tokenizer.model_max_length,truncation=True,).input_ids[0,2].item() for i in range(9)]
        numeric_index = torch.isin(input_ids, torch.tensor(numeric_tokens))
        targets[numeric_index] = IGNORE_INDEX
    return dict(
        input_ids=input_ids,
        labels=targets,
    )



def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN + '\n'
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    ignore_temporal: bool = False,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image, ignore_temporal=ignore_temporal)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)

@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        if not hasattr(self, 'iteration_step'):
            self.iteration_step = 0
        batch = self.get_batch(instances)
        if 'clip2' in instances[0]:
            batch['clip2']=self.get_batch([ins['clip2'] for ins in instances])
        batch['iteration_step'] = torch.tensor(self.iteration_step)
        self.iteration_step += 1
        return batch

    def get_batch(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            # for image in images:
            #     print('batch.image.shape', image.shape)
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
        if 'start_end_frame' in batch:
            batch['start_end_frame'] = torch.stack([torch.tensor(instance['start_end_frame']) for instance in instances])


        # print('batchimage.shape', batch['images'].shape)
        if 'query_feat' in instances[0]:
            query_feats = [instance['query_feat'] for instance in instances]
            batch['query_feats'] = pad_sequences_1d(query_feats, dtype=query_feats[0].dtype, fixed_length=None)
        if 'neg' in instances[0]:
            batch['neg'] = torch.stack([i['neg'] for i in instances])
        return batch

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments,
                 data_collator: DataCollatorForSupervisedDataset):
        super(LazySupervisedDataset, self).__init__()

        self.data_collator = data_collator
        self.tokenizer = tokenizer
        self.list_data_dict = json.load(open(data_path, "r"))
        if data_args.sparse_length>0:
            self.list_data_dict = [a for a in self.list_data_dict if a['meta']['duration'] > data_args.sparse_length]
        if data_args.sparse_dataset:
            temp = collections.defaultdict(list)
            for d in self.list_data_dict:
                temp[d['id']].append(d)
            temp = [random.choice(t) for t in list(temp.values())]
            self.list_data_dict = temp

        self.neg_value = 'no' if data_args.retrieval_only else 'Not Present'
        self.t2v = data_args.t2v
        if data_args.t2v is not None:
            with open(data_args.t2v, 'r') as f:
                dict_reader = DictReader(f)
                self.t2v = list(dict_reader)
                self.t2v = [mt['video_id'] for mt in self.t2v]
        if data_args.neg_window:
            if data_args.retrieval_only and self.t2v is None:
                for data in self.list_data_dict:
                    data['conversations'][0]['value'] = data['conversations'][0]['value'].replace('<video>\nDuring which frames can we see ', '')[:-1]
                    data['conversations'][0]['value'] = '<video>\nDoes {} happen in the video? Write your answer either yes or no.'.format(data['conversations'][0]['value'])
                    data['conversations'][1]['value'] = 'yes'
            # if data_args.hierarchy and self.t2v is None:
            #     for data in self.list_data_dict:
            #         data['conversations'][0]['value'] = data['conversations'][0]['value'].replace('During which frames can we see ', 'During which video can we see ')
            #         data['conversations'][1]['value'] = 'yes'
            if data_args.neg_samples > 1:
                for data in self.list_data_dict[::int(data_args.neg_samples)]:
                    data['conversations'][1]['value'] = self.neg_value
            else:
                neg_list = []
                for data in self.list_data_dict[::int(1/data_args.neg_samples)]:
                    neg_data = copy.deepcopy(data)
                    neg_data['conversations'][1]['value'] = self.neg_value
                    neg_list.append(neg_data)
                for i in range(data_args.neg_factor):
                    self.list_data_dict = self.list_data_dict + neg_list
        self.data_args = data_args
        self.appearance_visual_env = None
        self.textual_env = None
        self.videofeat = {}


    def _init_db(self):
        if self.data_args.vis_feat_storage == 'lmdb':
            self.appearance_visual_env = lmdb.open(self.data_args.feat_folder, readonly=True, create=False, max_readers=4096 * 8, readahead=False,lock=False)
            self.appearance_visual_txn = self.appearance_visual_env.begin(buffers=True)
        if self.data_args.q_feat_dir is not None:
            self.textual_env = lmdb.open(self.data_args.q_feat_dir, readonly=True, create=False, max_readers=4096 * 8, readahead=False)
            self.textual_txn = self.textual_env.begin(buffers=True)

    def _get_video_appearance_feat_by_vid(self, vid):
        if self.data_args.vis_feat_storage == 'lmdb':
            dump = self.appearance_visual_txn.get(vid.encode())
            with io.BytesIO(dump) as reader:
                img_dump = np.load(reader, allow_pickle=True)
                v_feat = img_dump['features'] if 'features' in img_dump else img_dump['memory_global']
        else:
            path = os.path.join(self.data_args.feat_folder, vid + '.npy')
            v_feat = np.load(path)
        # if self.normalize_v:
        #     _v_feat = l2_normalize_np_array(v_feat)
        # return torch.from_numpy(v_feat)  # (Lv, D)
        return v_feat

    def _get_query_feat_by_qid(self, qid, normalize_t=False):
        """
        qid: query_id
        returns both textual token feature and holistic text feature for each query
        """
        dump = self.textual_txn.get(qid.encode())
        with io.BytesIO(dump) as reader:
            q_dump = np.load(reader, allow_pickle=True)
            q_feat = q_dump['token_features']
            try:
                cls_q_feat = q_dump['cls_features']
            except:
                cls_q_feat = q_dump['eot_features']
            if len(cls_q_feat.shape) == 2:
                cls_q_feat = cls_q_feat[0]

        # if self.data_args.q_feat_type == "last_hidden_state":
        # q_feat = q_feat[:self.data_args.max_q_l]

        # if self.data_args.normalize_t:
        #     q_feat = self.l2_normalize_np_array(q_feat)

        cls_q_feat = self.l2_normalize_np_array(cls_q_feat)

        return torch.from_numpy(q_feat), cls_q_feat  # (Lq, D), (D, )

    def l2_normalize_np_array(self, np_array, eps=1e-5):
        """np_array: np.ndarray, (*, D), where the last dim will be normalized"""
        return np_array / (np.linalg.norm(np_array, axis=-1, keepdims=True) + eps)

    def __len__(self):
        return len(self.list_data_dict)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if self.t2v is None and self.data_args.hierarchy:
            if self.data_args.clip_adapter_feature=='alternate' and hasattr(self.data_collator, 'iteration_step'):
                if self.data_collator.iteration_step % 2 == 1:
                    return self.getitem(i)
            try:
                source = copy.deepcopy(self.list_data_dict[i])
                neg_images = []
                num_pos = random.randint(2,3)
                if self.data_args.hierarchy_zoom:
                    if self.data_args.fix_hierarchy_zoom > 0:
                        hierarchy_zoom = self.data_args.fix_hierarchy_zoom
                    else:
                        hierarchy_zoom = random.sample([1, 2], 1)[0]
                else:
                    hierarchy_zoom = 1
                if source['conversations'][1]['value'] == self.neg_value and self.data_args.hierarchy_neg:
                    num_neg_videos = self.data_args.hierarchy_num_videos//hierarchy_zoom
                else:
                    num_neg_videos = self.data_args.hierarchy_num_videos//hierarchy_zoom - num_pos

                starts = []
                while len(starts) < num_neg_videos:
                    neg_data = self.getitem(i, NEG=True)
                    # if neg_data['hier_neg_start'] in starts:
                    #     continue
                    if source['meta']['token']['<e0>'] < neg_data['hier_neg_start']/self.data_args.feature_fps or  \
                        source['meta']['token']['<s0>'] > neg_data['hier_neg_start']/self.data_args.feature_fps + self.data_args.debug_window:
                        neg_images.append(neg_data['image'])
                        starts.append(neg_data['hier_neg_start'])

                starts = numpy.array(starts)
                inds = starts.argsort()
                starts = [starts[i] for i in inds]
                neg_images = [neg_images[i] for i in inds]

                if source['conversations'][1]['value'] == self.neg_value and self.data_args.hierarchy_neg:
                    pos_data = neg_data
                    image = np.stack(neg_images, axis=0)
                else:
                    pos_idx = random.randint(0, self.data_args.hierarchy_num_videos//hierarchy_zoom - num_pos)
                    pos_data = []
                    for i in range(num_pos):
                        if hierarchy_zoom>1:
                            pos_data.append(self.getitem(i, conv_value = f"From {hierarchy_zoom*pos_idx} to {hierarchy_zoom*(pos_idx+num_pos-1)+1}."))#f"In video {pos_idx}")
                        else:
                            if self.data_args.hierarchy_zoom:
                                pos_data.append(self.getitem(i, conv_value = f"From {pos_idx} to {pos_idx+num_pos-1}."))#f"In video {pos_idx}")
                            else:
                                pos_data.append(self.getitem(i, conv_value = f"From {pos_idx} to {pos_idx+num_pos}."))
                    image = [pd['image'] for pd in pos_data]
                    image = neg_images[:pos_idx] + image + neg_images[pos_idx:]
                    image = [item for item in image for i in range(hierarchy_zoom)]
                    image = np.stack(image, axis=0)
                pos_data[0]['image'] = torch.from_numpy(image)
                return pos_data[0]
            except:
                if self.data_args.debug_my_dataset:
                    raise
                    print(e)
                return random.choice(self)

        if self.data_args.stream:
            toss1 = random.randint(0, 1) # toss=0 means negative
            toss2 = random.randint(0, 1)
            clip1 = self.getitem(i, NEG=toss1==0)
            clip2 = self.getitem(i, NEG=toss2==0, clip2=True)
            if clip2 is None:
                return random.choice(self)
            #if start1<start2:
            clip1['clip2']=clip2
            return clip1
            #else:
                #clip2['clip2']=clip1
                #return clip2
        return self.getitem(i)


    def getitem(self, i, NEG=False, clip2=False, conv_value=None) -> Dict[str, torch.Tensor]:
        source = copy.deepcopy(self.list_data_dict[i])

        data_type = 'video'
        if '<image>' in source['conversations'][0]['value']:
            source['conversations'][0]['value'] = source['conversations'][0]['value'].replace('<image>', '<video>')
            data_type = 'image'

        if clip2:
            source['conversations'][0]['value'] = source['conversations'][0]['value'] + '\n<memory>'

        # factor = random.randrange(1,5)
        # image = torch.zeros((self.data_args.num_frames if data_type == 'video' else 1, 768), dtype=torch.float16)
        # image = np.zeros((self.data_args.num_frames*factor if data_type == 'video' else 1, 768))

        try:
            if self.t2v is not None:
                if self.data_args.hierarchy:
                    neg_images = []
                    if source['conversations'][1]['value'] == self.neg_value:
                        num_videos = self.data_args.hierarchy_num_videos
                    else:
                        num_videos = self.data_args.hierarchy_num_videos - 1
                    neg_ids = random.sample([x for x in self.t2v if x != source['id']], num_videos)
                    for nid in neg_ids:
                        imagen = self._get_video_appearance_feat_by_vid(nid)
                        start, end = 0, imagen.shape[0] - 1
                        sampled_indices = np.linspace(start, end, self.data_args.num_frames, dtype=np.int32)
                        neg_images.append(imagen[sampled_indices])
                    if source['conversations'][1]['value'] == self.neg_value:
                        source['conversations'][1]['value'] = "Not Present"
                        image = np.stack(neg_images, axis=0)
                    else:
                        pos_idx = random.randint(0, self.data_args.hierarchy_num_videos-1)
                        source['conversations'][1]['value'] = f"In video {pos_idx}"
                        image = self._get_video_appearance_feat_by_vid(source['id'])
                        start, end = 0, image.shape[0] - 1
                        sampled_indices = np.linspace(start, end, self.data_args.num_frames, dtype=np.int32)
                        image = image[sampled_indices]
                        image = neg_images[:pos_idx] + [image] + neg_images[pos_idx:]
                        image = np.stack(image, axis=0)
                else:
                    if source['conversations'][1]['value'] == self.neg_value:
                        source['id'] = random.choice([x for x in self.t2v if x != source['id']])
                    image = self._get_video_appearance_feat_by_vid(source['id'])
                    start, end = 0, image.shape[0] - 1
                    sampled_indices = np.linspace(start, end, self.data_args.num_frames, dtype=np.int32)
                    image = image[sampled_indices]

                if self.data_args.q_feat_dir is not None:
                    if self.textual_env is None:
                        self._init_db()
                    query_feat, query_cls_feat = self._get_query_feat_by_qid(str(source["query_id"]))
            else:
                if self.data_args.dataset == 'mad':
                    if self.appearance_visual_env is None:
                        self._init_db()
                    if 'query_id' in source and self.data_args.q_feat_dir is not None:
                        query_feat, query_cls_feat = self._get_query_feat_by_qid(source["query_id"])
                    if self.data_args.vis_feat_storage == 'lmdb':
                        if source['id'] not in self.videofeat:
                            self.videofeat[source['id']] = self._get_video_appearance_feat_by_vid(source['id'])
                        image = self.videofeat[source['id']]
                    else:
                        image = self._get_video_appearance_feat_by_vid(source['id'])
                else:
                    if self.data_args.q_feat_dir is not None:
                        if self.textual_env is None:
                            self.textual_env = lmdb.open(self.data_args.q_feat_dir, readonly=True, create=False,
                                                         max_readers=4096 * 8, readahead=False)
                            self.textual_txn = self.textual_env.begin(buffers=True)
                        query_feat, query_cls_feat = self._get_query_feat_by_qid(source["id"])
                    if 'ego4d_stage1' in self.data_args.feat_folder:
                        feature_path = '{}/{}.pt'.format(self.data_args.feat_folder, source['id'])
                        # print('feature_path ', feature_path)
                        if not os.path.isfile(feature_path):
                            return random.choice(self)
                        image =  torch.load(feature_path)#.detach().cpu().numpy()
                    else:
                        feature_path = '{}/{}.npy'.format(self.data_args.feat_folder, source['id'])
                        image = np.load(feature_path) # <N, 768> float16

                if image.shape[0] < self.data_args.num_frames or len(image.shape)==1:
                    if clip2:
                        return None
                    return random.choice(self)
                if (self.data_args.clip_adapter or (data_type == 'video' and 'meta' in source)) and self.data_args.dataset == 'mad':
                    meta_start = source['meta']['token']['<s0>']
                    meta_end = source['meta']['token']['<e0>']
                    mad_feature_fps = self.data_args.feature_fps  # 30/16 if 'ego' in self.data_args.feat_folder else 5
                    change_fps=False
                    source_meta_duration = source['meta']['duration']
                    if (source['source'] == 'vidchapters7m' and source['meta']['duration'] < 2 * self.data_args.debug_window
                            and self.data_args.feature_fps==2):
                        change_fps = True
                        meta_start *=2
                        meta_end *=2
                        mad_feature_fps *=2
                        source_meta_duration *=2
                        source['meta']['token']['<s0>'] = meta_start
                        source['meta']['token']['<e0>'] = meta_end
                        source['meta']['duration'] = source_meta_duration
                    if source['source'] == 'stage2' or self.data_args.long_baseline:
                        mad_feature_fps = 1
                        meta_start *= 100/source_meta_duration
                        meta_end *= 100/source_meta_duration
                        source_meta_duration = 100
                        source['meta']['token']['<s0>'] = meta_start
                        source['meta']['token']['<e0>'] = meta_end
                        source['meta']['duration'] = source_meta_duration
                    if self.data_args.debug_window <= math.ceil(meta_end - meta_start) and source['source'] == 'stage2' and source['conversations'][1]['value'] == self.neg_value:
                            return random.choice(self)
                    if source['conversations'][1]['value'] == self.neg_value or NEG:
                        if meta_start > self.data_args.debug_window + 1:
                            toss = random.randint(0, 1)
                            if toss == 0 and meta_end < math.floor(source_meta_duration) - self.data_args.debug_window - 2 and not NEG:
                                meta_start = random.randint(math.ceil(meta_end) + 1, math.floor(
                                    source_meta_duration - self.data_args.debug_window - 1))
                            else:
                                meta_start = random.randint(0, math.floor(meta_start) - self.data_args.debug_window - 1)
                        else:
                            meta_start = random.randint(math.ceil(meta_end) + 1, math.floor(source_meta_duration -
                                                                                            self.data_args.debug_window -1))
                        meta_end = meta_start + 1

                    gt_len = math.ceil(meta_end - meta_start)
                    if self.data_args.debug_window <= gt_len and source['source'] != 'stage2':
                        if self.data_args.keep_longer_gt:
                            if random.randint(0, 1) == 0:
                                source['meta']['token']['<e0>'] = meta_start + self.data_args.debug_window - 1
                            else:
                                source['meta']['token']['<s0>'] = meta_end - self.data_args.debug_window + 1
                        else:
                            if clip2:
                                return None
                            return random.choice(self)

                    if self.data_args.debug_window != 0:
                        # print('\n', '<s0>:', source['meta']['token']['<s0>'],'<e0>:',  source['meta']['token']['<e0>'])
                        offset = random.randrange(self.data_args.debug_window - gt_len)
                        start_s = max(0, meta_start - offset)
                        end = start_s + self.data_args.debug_window
                        start, end = round(start_s*mad_feature_fps), round(end * mad_feature_fps)
                        # if source['conversations'][1]['value'] == self.neg_value and end > image.shape[0]-1:
                        #     print()
                        if end > image.shape[0]-1:
                            end = image.shape[0] - 1
                            start = max(0, end - round(self.data_args.debug_window * mad_feature_fps))
                            start_s = start/mad_feature_fps
                    else:
                        start, end = 0, image.shape[0]-1
                    # print('offset start_s, start, end:',offset, start_s, start, end)
                    if self.data_args.debug_window <= gt_len and source['source'] == 'stage2' and source['conversations'][1]['value'] != self.neg_value:
                        start, end = 0, image.shape[0] - 1
                    if image.shape[0] > self.data_args.num_frames:
                        sampled_indices = np.linspace(start, end, self.data_args.num_frames, dtype=np.int32)
                        image = image[sampled_indices]

            if not torch.is_tensor(image):#'ego4d_stage1' not in self.data_args.feat_folder:
                image = torch.from_numpy(image)

            if data_type == 'image' and len(image.shape) == 1:# <768>
                image = image.unsqueeze(0)

            if 'meta' in source:
                def convert(duration, x):
                    x = x / duration * self.data_args.num_frames
                    x = str(min(round(x), self.data_args.num_frames - 1))
                    if len(x) == 1:
                        x = "0" + x
                    return x
                if conv_value is not None:
                    source['conversations'][1]['value'] = conv_value
                    matches = re.search(r"(\d+) (to|and) (\d+)", conv_value)
                    from_number = int(matches.group(1))
                    to_number = int(matches.group(3))
                    replace_set = [('<s0>', str(from_number)), ('<e0>', str(to_number))]
                else:
                    replace_set = []
                    for k, v in source['meta']['token'].items():
                        if self.data_args.debug_window != 0:
                            duration = self.data_args.debug_window
                            if change_fps:
                                duration *= 2
                            v = source['meta']['token'][k] - start_s
                        else:
                            duration = source['meta']['duration']
                        replace_set.append((k, convert(duration, v)))

                    for l in range(len(source['conversations'])):
                        for x1, x2 in replace_set:
                            source['conversations'][l]['value'] = source['conversations'][l]['value'].replace(x1, x2)#DEFAULT_IGNORE_TOKEN if self.data_args.ignore_temporal else
                    # print(replace_set)
        except Exception as e:
            if self.data_args.debug_my_dataset:
                raise
                print(e)
            # else:
            #     if clip2:
            #         return None
            return random.choice(self)

        if NEG:
            source['conversations'][1]['value'] = self.neg_value

        if getattr(self.tokenizer, 'name', None) == 'GLMTokenizer':
            data_dict = preprocess_glm([source["conversations"]], self.tokenizer)
        else:
            data_dict = preprocess([source["conversations"]], self.tokenizer, has_image=True, ignore_temporal=self.data_args.ignore_temporal)
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0], labels=data_dict["labels"][0])
        if image.shape[0] not in [self.data_args.num_frames, self.data_args.hierarchy_num_videos] or not torch.is_tensor(image):
            if clip2:
                return None
            return random.choice(self)
        data_dict['image'] = image
        # print('dataset.image.shape', image.shape)
        if self.data_args.q_feat_dir is not None:
            data_dict['query_feat'] = query_feat
        if 'meta' in source:
            data_dict['start_end_frame'] = [int(x2) for x1, x2 in replace_set]
        if self.data_args.stream:
            data_dict['neg'] = torch.tensor(0 if NEG else 1)
            return data_dict
        data_dict["hier_neg_start"] = start
        return data_dict





def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args: DataArguments) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args,
                                data_collator=data_collator)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)
