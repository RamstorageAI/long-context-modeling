# coding=utf-8
# Copyright (c) 2024 Ant Group
# Author: Xiang Hu
import random
import json
import torch
import argparse
import sys
from torch.utils import data
from transformers import EvalPrediction, AutoTokenizer
from model.ramba_config import RambaConfig
from model.ramba_mixer_model import RambaLMHeadModel

from transformers import AutoConfig
from tqdm import tqdm
from model.model_factory import load_pretrained
from torch.utils.data import SequentialSampler
from reader.lazy_loader import LazyLoader
from reader.dataset import LongRNNDataset
from reader.ruler_collator import RulerSynthesizer
from flash_attn.losses.cross_entropy import CrossEntropyLoss


class Evaluator(object):
    def __init__(self, 
                 task_id,
                 model,
                 device):
        self.task_id = task_id
        self.model = model

        self.device = device

    def eval(self, 
             data_loader, 
             amp_dtype=torch.bfloat16):

        # total_step = sum(map(lambda x: len(x), data_loaders))

        epoch_iterator = tqdm(data_loader, desc="Iteration")
        self.model.eval()

        hits = 0
        steps = 0
        for inputs in epoch_iterator:
            steps += 1

            for k, v in inputs.items():
                if v is not None and isinstance(v, torch.Tensor):
                    inputs[k] = v.to(self.device)

            model_inputs = {"input_ids": inputs['input_ids']}
            input_ids = inputs['input_ids']
            with torch.amp.autocast('cuda', dtype=amp_dtype), torch.no_grad():
                result = self.model(input_ids, use_cache=True)

            labels_len = inputs['labels'].shape[1] - 1
            prediction = result.logits[:, -2 - labels_len: -2].argmax(dim=-1)
            labels = inputs['labels']

            hits += (prediction == labels[:, :-1]).all()

            if steps % 5 == 0:
                tqdm.write(str(hits / steps))
                # print(hits / steps)

        return hits / steps


def load_model(model, model_path, strict=True):
    state_dict = torch.load(model_path, map_location=lambda a, b: a)
    transfered_state_dict = {}
    for k, v in state_dict.items():
        new_k = k.replace('module.', '')
        transfered_state_dict[new_k] = v
    model.load_state_dict(transfered_state_dict, strict=strict)


if __name__ == '__main__':
    cmd = argparse.ArgumentParser('NCR pretraining setup')
    cmd.add_argument('--config_path', required=False, type=str, default=None)
    cmd.add_argument('--vocab_dir', required=True, type=str)
    cmd.add_argument('--corpus_path', required=True, type=str, help='path to the training corpus')
    cmd.add_argument('--model_type', type=str, required=True)
    cmd.add_argument('--task_id', type=int, default=0)
    cmd.add_argument('--max_seq_len', default=16384, type=int)
    cmd.add_argument('--checkpoint_path', required=False, type=str, help='directory of the checkpoints')
    cmd.add_argument('--peft_adapter_path', required=False, type=str)

    args = cmd.parse_args(sys.argv[1:])
    print(args)

    global_rank = -1
    device = torch.device('cpu')
    if torch.cuda.is_available():
        device = torch.device('cuda:0')

    model = load_pretrained(args.model_type, args.checkpoint_path, config_path=args.config_path, peft_adapter_path=args.peft_adapter_path)
    model.to(device, non_blocking=True)
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

    tokenizer = AutoTokenizer.from_pretrained(args.vocab_dir)
    data_collator = RulerSynthesizer(tokenizer, task_id=args.task_id)
    dataloader =  data.DataLoader(dataset,
                                  batch_size=1,
                                  collate_fn=data_collator.single_token_eval_collate_fn,
                                  sampler=SequentialSampler(dataset),
                                  num_workers=5
                                  )

    n_gpu = 1


    # force setting base learning rate
    # scheduler.base_lrs = [args.lr * args.accumulation_steps * lr_coeff, args.parser_lr * args.accumulation_steps * lr_coeff]
    
    evaluator = Evaluator(args.task_id, model, device=device)

    amp_dtype=torch.float16
    if torch.cuda.is_bf16_supported():
        amp_dtype=torch.bfloat16

    acc = evaluator.eval(dataloader, amp_dtype=amp_dtype)
    print(f'acc: {acc}')