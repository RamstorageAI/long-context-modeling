from transformers import Trainer
import torch
from torch.utils.data import SequentialSampler
from typing import Optional
from flash_attn.losses.cross_entropy import CrossEntropyLoss


class MambaTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # input_ids = inputs.pop("input_ids")
        input_ids = inputs["input_ids"]
        input_ids_ = torch.where(input_ids < 0, 0, input_ids)
        # model_input = {"input_ids": input_ids}
        # model_out = model(**model_input)
        model_out = model(input_ids_)
        lm_logits = model_out.logits.contiguous()
        # labels = model_out.labels.contiguous()

        # labels = torch.zeros_like(input_ids).fill_(-100)
        # labels[:, :-1] = input_ids[:, 1:]
        labels = torch.roll(input_ids, -1, 1)
        labels[:, -1] = -100

        loss_fct = CrossEntropyLoss(ignore_index=-100)
        lm_loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        return (lm_loss, lm_logits) if return_outputs else lm_loss

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        # return DistributedSampler(self.train_dataset, shuffle=False)
        return SequentialSampler(self.train_dataset)