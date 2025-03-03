import os
import random

import numpy as np
import torch
from transformers.modeling_utils import unwrap_model
from transformers import Trainer
from typing import Optional
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

from revisionllm.constants import PREFIX


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, 'no ignore status')
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return

# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, name=k) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

class VTimeLLMTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """
        if self.args.stream:
            neg = inputs.pop('neg')
            inputs2 = inputs.pop('clip2')
            n = inputs2.pop('neg')
            if self.args.stream_loss=='double':
                outputs = model(**inputs)
            if self.args.memory_type == 'mean_pool':
                inputs2['visual_memory'] = inputs['images'].mean(1)
            elif self.args.memory_type == 'gt_mean_pool':
                inputs2['visual_memory'] = []
                for i, image in enumerate(inputs['images']):
                    if neg[i]==0:
                        window = random.randint(5, 50)
                        start = random.randint(0, len(image) - window)
                        inputs2['visual_memory'].append(image[start:start+window].mean(0))
                    else:
                        inputs2['visual_memory'].append(
                            image[inputs['start_end_frame'][i][0]:inputs['start_end_frame'][i][1] + 1].mean(0))
                inputs2['visual_memory'] = torch.stack(inputs2['visual_memory'])
            elif self.args.memory_type == 'gt_multi':
                inputs2['visual_memory'] = []
                for i, image in enumerate(inputs['images']):
                    if neg[i] == 0: #NEG
                        window = random.randint(10, 50)
                        start = random.randint(0, len(image) - window - 1)
                        end = start+window
                    else:
                        start, end = (inputs['start_end_frame'][i][0].detach().cpu().item(),
                                    inputs['start_end_frame'][i][1].detach().cpu().item())
                    sampled_indices = np.linspace(start, end, 5, dtype=np.int32)
                    sampled_indices = torch.from_numpy(sampled_indices).to(torch.long)
                    inputs2['visual_memory'].append(image[sampled_indices])
                inputs2['visual_memory'] = torch.stack(inputs2['visual_memory'])
            elif self.args.memory_type == 'multi_pool':
                inputs2['visual_memory'] = []
                for i, image in enumerate(inputs['images']):
                    sampled_indices = np.linspace(0, inputs['images'].shape[1], 6, dtype=np.int32)
                    pool = []
                    for j in range(len(sampled_indices)-1):
                        pool.append(inputs['images'][i, sampled_indices[j]:sampled_indices[j+1]].mean(0))
                    inputs2['visual_memory'].append(torch.stack(pool))
                inputs2['visual_memory'] = torch.stack(inputs2['visual_memory'])
            elif self.args.memory_type == 'mmllm_pool':
                raise NotImplementedError

            inputs2['prefix_memory'] =  torch.stack([self.tokenizer(PREFIX[n],
                                                        return_tensors="pt",
                                                        padding="longest",
                                                        max_length=self.tokenizer.model_max_length,
                                                        truncation=True).input_ids for n in neg.tolist()]).to(inputs2['input_ids']).squeeze()
            outputs2 = model(**inputs2)
            if self.args.stream_loss=='double':
                loss = self.get_loss(inputs, model, outputs)
                loss2 = self.get_loss(inputs2, model, outputs2)
                loss = loss+loss2
            else:
                loss = self.get_loss(inputs2, model, outputs2)
            return (loss, outputs2) if return_outputs else loss
        else:
            outputs = model(**inputs)
            loss = self.get_loss(inputs, model, outputs)
            return (loss, outputs) if return_outputs else loss

    def get_loss(self, inputs, model, outputs):
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]
        if labels is not None:
            if unwrap_model(model)._get_name() in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        return loss

    def _save_checkpoint(self, model, trial, metrics=None):
        # if self.args.lora_enable:
        # from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
        # checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"
        # run_dir = self._get_output_dir(trial=trial)
        # output_dir = os.path.join(run_dir, checkpoint_folder)
        # state_dict = get_peft_state_maybe_zero_3(
        #     self.model.named_parameters(), self.args.lora_bias
        # )
        # non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
        #     self.model.named_parameters()
        # )
        # if self.args.local_rank == 0 or self.args.local_rank == -1:
        #     self.model.config.save_pretrained(output_dir)
        #     self.model.save_pretrained(output_dir, state_dict=state_dict)
        #     torch.save(non_lora_state_dict, os.path.join(output_dir, 'non_lora_trainables.bin'))
#         elif getattr(self.args, 'tune_mm_mlp_adapter', False):
#             from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
#             checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

#             run_dir = self._get_output_dir(trial=trial)
#             output_dir = os.path.join(run_dir, checkpoint_folder)

#             # Only save Adapter
#             keys_to_match = ['mm_projector']
#             if getattr(self.args, "use_im_start_end", False):
#                 keys_to_match.extend(['embed_tokens', 'embed_in'])

#             weight_to_save = get_mm_adapter_state_maybe_zero_3(self.model.named_parameters(), keys_to_match)

#             if self.args.local_rank == 0 or self.args.local_rank == -1:
#                 self.model.config.save_pretrained(output_dir)
#                 torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        # else:
        super(VTimeLLMTrainer, self)._save_checkpoint(model, trial, metrics)

    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        if getattr(self.args, 'tune_mm_mlp_adapter', False):
            pass
        else:
            super(VTimeLLMTrainer, self)._save(output_dir, state_dict)
