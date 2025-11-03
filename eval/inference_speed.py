from unittest import TestCase
from model.model_factory import load_pretrained
import torch
import torch.nn.functional as F
import random
import argparse
import math
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm
import pickle
import os
import sys
from transformers import AutoConfig, AutoTokenizer
from reader.lazy_loader import LazyChunkedLoader, LazyLoader
from reader.dataset_new import TextDataset
import time


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    cmd = argparse.ArgumentParser('qa gen setup')
    cmd.add_argument('--ckpt_path', type=str, required=True)
    cmd.add_argument('--model_type', type=str, required=True)
    cmd.add_argument('--config_path', type=str, default=None)
    cmd.add_argument('--vocab_dir', type=str, default='configs/gpt2-neox-20b')
    cmd.add_argument('--corpus_path', type=str, required=True)
    cmd.add_argument('--max_input_len', type=int, required=True)
    cmd.add_argument('--gen_tokens', type=int, default=100)
    cmd.add_argument('--batch_size', type=int, default=32)
    cmd.add_argument('--test_num', type=int, default=6)
    args = cmd.parse_args(sys.argv[1:])

    set_seed(1)
    model = load_pretrained(args.model_type, args.ckpt_path, args.config_path)
    device = torch.device('cuda:0')
    print (args)
    if args.max_input_len:
        print (f"input length is {args.max_input_len}")
    model.to(device)

    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(args.vocab_dir)

    valid_ds = LazyLoader(args.corpus_path, array_data_type=np.uint16)
    valid_dataset = TextDataset(
        valid_ds,
        batch_size=args.batch_size,
        segment_len=args.max_input_len,
        num_samples = -1,
        ramdom_sampling=False,
        epochs=1,
        is_lazy=True
    )


    def generate(model, **kwargs):
        return model.generate(**kwargs)
    
    dtype = torch.bfloat16

    num_test = args.test_num
    times = []
    for i in range(num_test):
        input_ids_batch = []


        batch_size = args.batch_size
        for j in range(batch_size):
            input_ids = valid_dataset[batch_size*i+j][0]
            input_ids_batch.append(input_ids)

        input_ids_batch = torch.stack(input_ids_batch)  # (batch_size, seq_length)
        input_ids_batch = input_ids_batch.to(device)
        print (f"input_ids_batch: {input_ids_batch.shape}")

        torch.cuda.synchronize()
        start_time = time.time()
        with torch.amp.autocast('cuda', dtype=dtype):
            out_ids = np.array([0])
            outputs = model.generate(
                input_ids=input_ids_batch, 
                max_new_tokens=args.gen_tokens + 1,
                do_sample=False, 
                use_cache=True,
                eos_token_id=None
            )
        torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
        out_ids = outputs.cpu().numpy()

        print (f"out_ids\n{out_ids.shape}")
    
    gen_max_len_token_time = sum(times[1:]) / (num_test - 1)

    times = []
    for i in range(num_test):
        input_ids_batch = []


        batch_size = args.batch_size
        for j in range(batch_size):
            input_ids = valid_dataset[batch_size*i+j][0]
            input_ids_batch.append(input_ids)

        input_ids_batch = torch.stack(input_ids_batch)  # 形状为 (batch_size, seq_length)
        input_ids_batch = input_ids_batch.to(device)
        print (f"input_ids_batch: {input_ids_batch.shape}")

        torch.cuda.synchronize()
        start_time = time.time()
        with torch.amp.autocast('cuda', dtype=dtype):
            out_ids = np.array([0])
            outputs = model.generate(
                input_ids=input_ids_batch, 
                max_new_tokens=1,
                do_sample=False, 
                use_cache=True,
                eos_token_id=None
            )
        torch.cuda.synchronize()
        end_time = time.time()
        times.append(end_time - start_time)
        out_ids = outputs.cpu().numpy()

        print (f"out_ids\n{out_ids.shape}")

    prefilling_time = sum(times[1:]) / (num_test - 1)

    print (f"gen {args.gen_tokens} tokens time: {gen_max_len_token_time - prefilling_time}")
