# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
CONFIG = {
  "_name_or_path": "/nfs/data3/user/models--lmsys--fastchat-t5-3b-v1.0/snapshots/0b1da230a891854102d749b93f7ddf1f18a81024",
  "architectures": [
    "T5ForConditionalGeneration"
  ],
  "bos_token_id": 1,
  "d_ff": 5120,
  "d_kv": 64,
  "d_model": 2048,
  "decoder_start_token_id": 0,
  "dense_act_fn": "gelu_new",
  "dropout_rate": 0.1,
  "eos_token_id": 1,
  "feed_forward_proj": "gated-gelu",
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_factor": 1.0,
  "initializer_range": 0.02,
  "intermediate_size": 11008,
  "is_encoder_decoder": True,
  "is_gated_act": True,
  "layer_norm_epsilon": 1e-06,
  "max_position_embeddings": 2048,
  "model_type": "VTimeLLM",
  "n_positions": 512,
  "num_attention_heads": 32,
  "num_decoder_layers": 24,
  "num_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 32,
  "num_layers": 24,
  "output_past": True,
  "pad_token_id": 0,
  "pretraining_tp": 1,
  "relative_attention_max_distance": 128,
  "relative_attention_num_buckets": 32,
  "rms_norm_eps": 1e-06,
  "rope_scaling": None,
  "task_specific_params": {
    "summarization": {
      "early_stopping": True,
      "length_penalty": 2.0,
      "max_length": 200,
      "min_length": 30,
      "no_repeat_ngram_size": 3,
      "num_beams": 4,
      "prefix": "summarize: "
    },
    "translation_en_to_de": {
      "early_stopping": True,
      "max_length": 300,
      "num_beams": 4,
      "prefix": "translate English to German: "
    },
    "translation_en_to_fr": {
      "early_stopping": True,
      "max_length": 300,
      "num_beams": 4,
      "prefix": "translate English to French: "
    },
    "translation_en_to_ro": {
      "early_stopping": True,
      "max_length": 300,
      "num_beams": 4,
      "prefix": "translate English to Romanian: "
    }
  },
  "tie_word_embeddings": False,
  "torch_dtype": "float16",
  "transformers_version": "4.31.0",
  "use_cache": True,
  "vocab_size": 32110
}

import os

from transformers import AutoModel, AutoTokenizer, PretrainedConfig

from revisionllm.model.vtimellm_llama import VTimeLLMConfig

root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")
from dataclasses import dataclass, field
import logging
import pathlib
from typing import Dict, Optional, Sequence, List

import torch
import transformers
import sys
sys.path.append(root_dir)
from revisionllm import conversation as conversation_lib
from revisionllm.train.vtimellm_trainer import VTimeLLMTrainer, get_peft_state_maybe_zero_3, \
    get_peft_state_non_lora_maybe_zero_3
from revisionllm.train.dataset import make_supervised_data_module, DataArguments
from revisionllm.model import VTimeLLMLlamaForCausalLM, VTimeLLMChatGLMForCausalLM
from revisionllm.model.builder import load_lora
from revisionllm.mm_utils import print_trainable_parameters

local_rank = None

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="lmsys/vicuna-7b-v1.5")
    stage2_path: Optional[str] = field(default=None)
    version: Optional[str] = field(default="v0")
    tune_mm_mlp_adapter: bool = field(default=False)
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    pretrain_clip_adapter: Optional[str] = field(default=None)
    mlp_adapter: bool = field(default=False)
    ca_adapter: bool = field(default=False)
    cross_attn: bool = field(default=False)
    self_attn: Optional[str] = field(default=None)
    ca_self_attn: Optional[str] = field(default=None)
    max_seq_length: int = field(default=2048)
    sa_pos: int = field(default=2)
    adapter_input_dim: int = field(default=768)
    debug_server: bool = field(default=False)
    clip_adapter: bool = field(default=False)
    clip_adapter_text: bool = field(default=False)
    clip_adapter_feature: Optional[str] = field(default='temporal') #'cls', "all", "alternate"
    hierarchy: bool = field(default=False)
    dual_adapter: bool = field(default=False)


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    training_stage: int = field(default=2)
    n_gpu: int = field(default=1)
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    memory_type: str = 'mean_pool' # gt_mean_pool, gt_multi_pool
    stream_loss: str= 'single' # double
    stage1_load_lora: bool = field(default=False)


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.stream = data_args.stream
    data_args.clip_adapter = model_args.clip_adapter
    data_args.clip_adapter_feature = model_args.clip_adapter_feature
    data_args.hierarchy = model_args.hierarchy
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    # if model_args.debug_server:
    #     kwargs = {'_from_auto': False, '_from_pipeline': None, 'proxies': None, 'resume_download': False, 'return_unused_kwargs': True, 'subfolder': ''}
    #     config, _ = PretrainedConfig.from_pretrained(pretrained_model_name_or_path=model_args.model_name_or_path,
    #                                                 cache_dir = None,
    #                                                 force_download = False,
    #                                                 local_files_only = False,
    #                                                 token = None,revision = 'main',**kwargs)
    #     config.name_or_path = model_args.model_name_or_path
    #     #config.quantization_config = quantization_config
    #     for k,v in CONFIG.items():
    #         setattr(config, k, v)
    #     model = VTimeLLMLlamaForCausalLM(config=config)
    # else:
    if 'chatglm' in model_args.model_name_or_path:
        model = VTimeLLMChatGLMForCausalLM.from_pretrained(
            model_args.model_name_or_path, empty_init=False, device='cuda'
        )
    elif 'vicuna' in model_args.model_name_or_path or 'longchat' in model_args.model_name_or_path or 'fastchat' in model_args.model_name_or_path:
        model = VTimeLLMLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args,
            low_cpu_mem_usage=model_args.debug_server,
            #use_safetensors=True,
        )
    else:
        model = AutoModel.from_pretrained(model_args.model_name_or_path)

    model.config.use_cache = False

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)

        # print_trainable_parameters(model)
        #if not model_args.debug_server:
        if training_args.training_stage == 1 and training_args.stage1_load_lora:
            model.get_model().initialize_vision_modules(model_args)
            model = load_lora(model, model_args.stage2_path)
            rank0_print('Merging LoRA weights...')
            # model = model.merge_and_unload()
        elif training_args.training_stage == 4:
            model.get_model().initialize_vision_modules(model_args)
            rank0_print('Finetune LoRA weights...')
            model = load_lora(model, model_args.stage2_path, is_trainable=True)
        elif training_args.training_stage == 3:
            model.get_model().initialize_vision_modules(model_args)
            model = load_lora(model, model_args.stage2_path)
            rank0_print('Merging LoRA weights...')
            model = model.merge_and_unload()
            # print_trainable_parameters(model)
            rank0_print("Adding LoRA adapters...")
            model = get_peft_model(model, lora_config)
        else:
            rank0_print("Adding LoRA adapters...")
            model = get_peft_model(model, lora_config)
        # print_trainable_parameters(model)

    if 'chatglm' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            trust_remote_code=True
        )
    elif 'vicuna' in model_args.model_name_or_path or 'longchat' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

 
    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter

    #if not model_args.debug_server:
    if training_args.training_stage != 3:
        model.get_model().initialize_vision_modules(model_args=model_args)

        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True
            if model_args.cross_attn:
                for p in model.get_model().cross_attn.parameters():
                    p.requires_grad = True

        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False
            if model_args.cross_attn:
                for p in model.get_model().cross_attn.parameters():
                    p.requires_grad = False

    if training_args.bits in [4, 8] or model_args.ca_adapter or model_args.clip_adapter:
        model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

    if model_args.cross_attn:
        model.get_model().cross_attn.to(dtype=compute_dtype, device=training_args.device)


    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    print_trainable_parameters(model)
    trainer = VTimeLLMTrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)

    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
