from transformers import Trainer
import torch
from torch.utils.data import DistributedSampler, SequentialSampler
from typing import Optional
import os
from flash_attn.losses.cross_entropy import CrossEntropyLoss


class MambaPasskeyTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # input_ids = inputs.pop("input_ids")
        input_ids = inputs["input_ids"]
        model_input = {"input_ids": input_ids}
        if 'pass_init_state' in inputs:
            model_input['pass_init_state'] = inputs['pass_init_state']
        outputs = model(**model_input)
        lm_logits = outputs.logits
        labels = outputs.labels

        # labels = torch.zeros_like(input_ids).fill_(-100)
        # labels[:, :-1] = input_ids[:, 1:]

        loss_fct = CrossEntropyLoss(ignore_index=-100)
        lm_loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        final_pos = inputs['final_pos']

        # reshape outputs
        if isinstance(final_pos, list):
            total_size = len(final_pos)
            group_num = len(list(filter(lambda x: x != -1, final_pos)))
            for _val in final_pos:
                if _val != -1:
                    final_pos = _val
                    break
            lm_logits = lm_logits.reshape(group_num, -1, lm_logits.shape[-1])
            input_ids = input_ids.reshape(group_num, -1)
        if input_ids.shape[1] == lm_logits.shape[1]:
            # no lmk inserted
            # print(f'no lmk inserted')
            return (lm_loss, {"logits": lm_logits[:, final_pos-1].argmax(dim=-1)}) if return_outputs else lm_loss
        else:
            chunk_num = lm_logits.shape[1] - input_ids.shape[1]
            assert input_ids.shape[1] % chunk_num == 0
            chunk_size = input_ids.shape[1] // chunk_num
            final_pos_ = (final_pos // chunk_size) * (chunk_size + 1) + final_pos % chunk_size + 1
            assert (final_pos_ - 1) % chunk_size != 0
            # print(f'final pos: {final_pos}, {final_pos_}')
            return (lm_loss, {"logits": lm_logits[:, final_pos_-1].argmax(dim=-1)}) if return_outputs else lm_loss

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        # return DistributedSampler(self.train_dataset, shuffle=False)
        return SequentialSampler(self.train_dataset)

