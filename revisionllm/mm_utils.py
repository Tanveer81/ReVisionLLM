from PIL import Image
from io import BytesIO
import base64
import numpy as np
import torch
import decord
# import cv2
from decord import gpu
from transformers import StoppingCriteria
from revisionllm.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_MEMORY_TOKEN, MEMORY_TOKEN_INDEX, \
    DEFAULT_IGNORE_TOKEN, IGNORE_INDEX


def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


def process_images(images, image_processor, model_cfg):
    return image_processor(images, return_tensors='pt')['pixel_values']


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    image_chunks = prompt.split(DEFAULT_IMAGE_TOKEN)
    if DEFAULT_MEMORY_TOKEN in image_chunks[1]:
        prompt_chunks = []
        prompt_chunks.append(tokenizer(image_chunks[0]).input_ids)
        #prompt_chunks.append(image_chunks[0])
        memory_chunks = image_chunks[1].split(DEFAULT_MEMORY_TOKEN)
        for mc in memory_chunks:
            prompt_chunks.append(tokenizer(mc).input_ids)
            #prompt_chunks.append(mc)
    # elif DEFAULT_IGNORE_TOKEN in image_chunks[1]:
    #     prompt_chunks = []
    #     prompt_chunks.append(tokenizer(image_chunks[0]).input_ids)
    #     #prompt_chunks.append(image_chunks[0])
    #     memory_chunks = image_chunks[1].split(DEFAULT_IGNORE_TOKEN)
    #     for mc in memory_chunks:
    #         prompt_chunks.append(tokenizer(mc).input_ids)
    #         #prompt_chunks.append(mc)
    else:
        prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split(DEFAULT_IMAGE_TOKEN)]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])
    elif hasattr(tokenizer, 'name') and tokenizer.name == "GLMTokenizer":
        offset = 2
        input_ids = prompt_chunks[0][:2]

    if DEFAULT_MEMORY_TOKEN in image_chunks[1]:
        for x in insert_separator(prompt_chunks[:2], [image_token_index] * (offset + 1)):
            input_ids.extend(x[offset:])
        input_ids.extend([MEMORY_TOKEN_INDEX])
        input_ids.extend(prompt_chunks[2])
    # elif DEFAULT_IGNORE_TOKEN in image_chunks[1]:
    #     for x in insert_separator(prompt_chunks[:2], [image_token_index] * (offset + 1)):
    #         input_ids.extend(x[offset:])
    #     input_ids.extend([IGNORE_INDEX])
    #     input_ids.extend(prompt_chunks[2])
    #     input_ids.extend([IGNORE_INDEX])
    #     input_ids.extend(prompt_chunks[3])
    else:
        for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
            input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]




class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        assert output_ids.shape[0] == 1, "Only support batch size 1 (yet)"  # TODO
        offset = min(output_ids.shape[1] - self.start_len, 3)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if output_ids[0, -keyword_id.shape[0]:].equal(keyword_id):
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        # print(_, param.requires_grad, param.numel())
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )

class VideoExtractor():
    """Dataset for supervised fine-tuning."""

    def __init__(self, N=100):
        self.N = N

    def extract(self, data, start_end=None, sample_fps=0):
        video_path = data['video']
        id = data['id']
        # try:
        video_reader = decord.VideoReader(video_path, num_threads=1)
        if start_end is None:
            total_frames = len(video_reader)
            start = 0
            end = total_frames - 1
        else:
            start = int(start_end[0])# * video_reader.get_avg_fps())
            end = int(start_end[1])#min(int(start_end[1] * video_reader.get_avg_fps()), len(video_reader)) - 1
            total_frames = end-start+1
        fps = video_reader.get_avg_fps()
        split = data.get('split', None)
        if split is not None:
            start = max(int(fps * split[0]), 0)
            end = min(int(fps * split[1]), total_frames - 1)
        if sample_fps > 0:
            sampled_indices = np.linspace(start, end, int((total_frames * sample_fps) // fps), dtype=np.int32)

        else:
            sampled_indices = np.linspace(start, end, self.N, dtype=np.int32)
        
#         sampled_frames = []
#         cap = cv2.VideoCapture(video_path)
#         for s in sampled_indices:
#             cap.set(1, s)
#             _, img = cap.read()  # read one frame from the 'capture' object; img is (H, W, C)
#             sampled_frames.append(img)
#         cap.release()
#         # cv2.destroyAllWindows()
#         sampled_frames = np.stack(sampled_frames, axis=0)  # dimensions (T, H, W, C)
        
        video_reader.skip_frames(1)
        sampled_frames = video_reader.get_batch(sampled_indices).asnumpy()
        
        # except Exception as e:
        #     print(e)
        #     return None, torch.zeros(1)
        
        images = torch.from_numpy(sampled_frames.transpose((0, 3, 1, 2)))
        return id, images, sampled_indices