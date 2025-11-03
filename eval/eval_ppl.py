# coding=utf-8
# Copyright (c) 2024 Ant Group
# Author: Xiang Hu
import random
import json
import torch
import argparse
import sys
from transformers import AutoConfig
from torch.utils import data
from tqdm import tqdm
from model.model_factory import load_pretrained
from torch.utils.data import SequentialSampler
from reader.lazy_loader import LazyLoader
from reader.dataset import LongRNNDataset
from reader.data_collator import LongRNNDataCollator, BOSLongRNNDataCollator
from flash_attn.losses.cross_entropy import CrossEntropyLoss


class Evaluator(object):
    def __init__(self, 
                 model,
                 device):
        self.model = model

        self.device = device

    def eval(self, 
             data_loader, 
             amp_dtype=torch.bfloat16):

        # total_step = sum(map(lambda x: len(x), data_loaders))

        epoch_iterator = tqdm(data_loader, desc="Iteration")
        self.model.eval()

        total_ppl = 0
        steps = 0
        for inputs in epoch_iterator:
            steps += 1

            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.device)

            with torch.amp.autocast('cuda', dtype=amp_dtype), torch.no_grad():
                result = self.model(**inputs, use_cache=True)
            
            input_ids = inputs["input_ids"]
            lm_logits = result.logits
            labels = result.labels
            # labels = torch.zeros_like(input_ids).fill_(-100)
            # labels[:, :-1] = input_ids[:, 1:]

            loss_fct = CrossEntropyLoss(ignore_index=-100)
            if lm_logits.shape[1] == labels.shape[1]:
                lm_loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            else:
                print(f'eval final tokens')
                labels = labels[:, -lm_logits.shape[1]:]
                lm_loss = loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))
            total_ppl += lm_loss
            if steps % 50 == 0:
                print(f'ppl: {total_ppl / steps}')

        return total_ppl / steps


def load_model(model, model_path, strict=True):
    state_dict = torch.load(model_path, map_location=lambda a, b: a)
    transfered_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace('module.', '')
        transfered_state_dict[new_k] = v
    model.load_state_dict(transfered_state_dict, strict=strict)


if __name__ == '__main__':
    cmd = argparse.ArgumentParser('NCR pretraining setup')
    cmd.add_argument('--config_path', type=str, help='config for ramba', default=None)
    cmd.add_argument('--corpus_path', required=True, type=str, help='path to the training corpus')
    cmd.add_argument('--add_bos', action='store_true')
    cmd.add_argument('--model_type', required=True, type=str)
    cmd.add_argument('--max_seq_len', default=16384, type=int)
    cmd.add_argument('--checkpoint_path', required=False, type=str, help='directory of the checkpoints')

    # torch.set_printoptions(profile='full')
    args = cmd.parse_args(sys.argv[1:])
    print(args)

    global_rank = -1
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:0')

    # with open(args.config_path, "r") as f_in:
    #     config_data = json.load(f_in)

    print(f'model type: {args.model_type}')
    

    model = load_pretrained(args.model_type, args.checkpoint_path, config_path=args.config_path)
    model.to(device)
    model.eval()

    valid_ds = LazyLoader(args.corpus_path)
    dataset = LongRNNDataset(
        valid_ds,
        batch_size=1,
        segment_len=args.max_seq_len,
        ramdom_sampling=False,
        segment_size=1,
        epochs=1
    )

    if not args.add_bos:
        print('not add bos')
        data_collator = LongRNNDataCollator()
    else:
        print('add bos')
        data_collator = BOSLongRNNDataCollator()
    dataloader =  data.DataLoader(dataset,
                                  batch_size=1,
                                  collate_fn=data_collator.ramba_collator_fn,
                                  sampler=SequentialSampler(dataset),
                                  num_workers=1
                                  )

    n_gpu = 1


    # force setting base learning rate
    # scheduler.base_lrs = [args.lr * args.accumulation_steps * lr_coeff, args.parser_lr * args.accumulation_steps * lr_coeff]
    
    evaluator = Evaluator(model, device=device)

    amp_dtype=torch.float16
    if torch.cuda.is_bf16_supported():
        amp_dtype=torch.bfloat16

    ppl = evaluator.eval(dataloader, amp_dtype=amp_dtype)
    print(f'perplexity: {ppl}')