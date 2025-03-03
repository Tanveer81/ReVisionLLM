import os
root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
import sys
sys.path.append(root_dir)
import argparse
import torch
from revisionllm.constants import IMAGE_TOKEN_INDEX
from revisionllm.conversation import conv_templates, SeparatorStyle
from revisionllm.model.builder import load_pretrained_model, load_lora
from revisionllm.utils import disable_torch_init
from revisionllm.mm_utils import tokenizer_image_token, KeywordsStoppingCriteria, VideoExtractor
from PIL import Image
import requests
from io import BytesIO
from transformers import TextStreamer
from easydict import EasyDict as edict
try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    from PIL import Image
    BICUBIC = Image.BICUBIC
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize
import numpy as np
import clip


def inference(model, image, query_feats, query, tokenizer, visual_memory = None, prefix_memory=None, return_list=False):
    if visual_memory is not None:
        query = query + '<memory>'
    conv = conv_templates["v1"].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
    input_ids = input_ids.repeat(image.shape[0], 1)
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        model_output = model.generate(
            input_ids,
            images=image.cuda() if torch.cuda.is_available() else image,
            query_feats = query_feats,
            do_sample=True,
            temperature=0.05,
            num_beams=1,
            # no_repeat_ngram_size=3,
            max_new_tokens=1024,
            use_cache=True,
            visual_memory=visual_memory,
            prefix_memory=prefix_memory,
            output_scores=True,
            return_dict_in_generate=True,
            output_hidden_states=True)

        # https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py#L1295
    output_ids = model_output['sequences']
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
    for i in range(len(outputs)):
        outputs[i] = outputs[i].strip()
        if outputs[i].endswith(stop_str):
            outputs[i] = outputs[i][:-len(stop_str)]
        outputs[i] = outputs[i].strip()
    if len(outputs)==1 and not return_list:
        outputs = outputs[0]
    return outputs, model_output


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--clip_path", type=str, default="/home/stud/user/VTimeLLM/checkpoints/clip/ViT-L-14.pt")
    parser.add_argument("--model_base", type=str, default="/home/stud/user/.cache/huggingface/hub/models--lmsys--vicuna-7b-v1.5/snapshots/de56c35b1763eaae20f4d60efd64af0a9091ebe5")
    parser.add_argument("--pretrain_mm_mlp_adapter", type=str, default="/home/stud/user/VTimeLLM/checkpoints/revisionllm-vicuna-v1-5-7b-stage1/mm_projector.bin")
    parser.add_argument("--stage2", type=str, default="/home/stud/user/VTimeLLM/checkpoints/revisionllm-vicuna-v1-5-7b-stage2")
    parser.add_argument("--stage3", type=str, default="/home/stud/user/VTimeLLM/checkpoints/revisionllm-vicuna-v1-5-7b-stage3")
    parser.add_argument("--video_path", type=str, default="/home/stud/user/0.mp4")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()
    disable_torch_init()
    tokenizer, model, context_len = load_pretrained_model(args, args.stage2, args.stage3)
    if torch.cuda.is_available():
        model = model.cuda()
    # model.get_model().mm_projector.to(torch.float16)
    model.to(torch.float32)

    clip_model, _ = clip.load(args.clip_path)
    clip_model.eval()
    if torch.cuda.is_available():
        clip_model = clip_model.cuda()

    video_loader = VideoExtractor(N=100)
    _, images = video_loader.extract({'id': None, 'video': args.video_path})

    transform = Compose([
        Resize(224, interpolation=BICUBIC),
        CenterCrop(224),
        Normalize((0.48145466, 0.4578275, 0.40821073),
                  (0.26862954, 0.26130258, 0.27577711)),
    ])

    # print(images.shape) # <N, 3, H, W>
    images = transform(images / 255.0)
    images = images.to(torch.float32)
    with torch.no_grad():
        features = clip_model.encode_image(images.to('cuda')
                    if torch.cuda.is_available() else images).to(torch.float32)

    query = "describe the video."
    print("query: ", query)
    print("answer: ", inference(model, features, "<video>\n " + query, tokenizer))

def inference_stage1(model, image, query, tokenizer):
    conv = conv_templates["v1"].copy()
    conv.append_message(conv.roles[0], query)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0)
    input_ids = input_ids.repeat(image.shape[0], 1)
    if torch.cuda.is_available():
        input_ids = input_ids.cuda()

    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image.cuda() if torch.cuda.is_available() else image,
            query_feats = None,
            do_sample=True,
            temperature=0.05,
            num_beams=1,
            # no_repeat_ngram_size=3,
            max_new_tokens=1024,
            use_cache=True,
            visual_memory=None,
            prefix_memory=None)

        # https://github.com/huggingface/transformers/blob/main/src/transformers/generation/utils.py#L1295

    input_token_len = input_ids.shape[1]
    n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
    if n_diff_input_output > 0:
        print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
    outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)
    for i in range(len(outputs)):
        outputs[i] = outputs[i].strip()
        if outputs[i].endswith(stop_str):
            outputs[i] = outputs[i][:-len(stop_str)]
        outputs[i] = outputs[i].strip()
    return outputs
