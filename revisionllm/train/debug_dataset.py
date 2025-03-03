import os
from revisionllm.train.train import ModelArguments, TrainingArguments
root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
from tqdm import tqdm
import torch
import transformers
import sys
sys.path.append(root_dir)
from revisionllm import conversation as conversation_lib
from revisionllm.train.dataset import make_supervised_data_module, DataArguments

def debug():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.stream = data_args.stream
    data_args.clip_adapter = model_args.clip_adapter
    data_args.hierarchy = model_args.hierarchy
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )
    tokenizer.pad_token = tokenizer.unk_token
    data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    a = data_module['train_dataset']
    for a in tqdm(torch.utils.data.DataLoader(data_module['train_dataset'])):
        b = 1

    print()


if __name__ == "__main__":
    debug()
