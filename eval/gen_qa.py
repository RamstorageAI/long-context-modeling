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
from reader.dataset_qa import QADataset
import os
import sys
from transformers import AutoConfig, AutoTokenizer


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
    cmd.add_argument('--output_path', type=str, required=True)
    cmd.add_argument('--max_input_len', type=int, default=8192)
    cmd.add_argument('--save_steps', type=int, default=100)
    args = cmd.parse_args(sys.argv[1:])

    set_seed(1)
    model = load_pretrained(args.model_type, args.ckpt_path, args.config_path)
    device = torch.device('cuda:0')
    model.to(device)
    model.bfloat16()
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(args.vocab_dir)

    dataset = QADataset(
        args.corpus_path, 
        tokenizer=tokenizer,
        eos_id=tokenizer.eos_token_id
    )

    def generate(model, **kwargs):
        return model.generate(**kwargs)

    output_ids = []
    if os.path.exists(args.output_path):
        with open(args.output_path, 'rb') as file_in:
            output_ids = pickle.load(file_in)
    print(f'output ids len: {len(output_ids)}')
    
    epoch_iterator = tqdm(dataset, desc="Iteration")
    for step, inputs in enumerate(epoch_iterator):
        if step < len(output_ids):
            continue
        # input_ids = torch.tensor(inputs['text'], device=device).unsqueeze(0)
        prompt_ids = np.concatenate([inputs['context'], inputs['question']])
        input_ids = torch.tensor(prompt_ids, device=device, dtype=torch.long).unsqueeze(0)
        dtype = torch.bfloat16
        with torch.amp.autocast('cuda', dtype=dtype):
            out_ids = np.array([0])
            outputs = generate(
                model, 
                input_ids=input_ids, 
                max_new_tokens=32,
                do_sample=False, 
                use_cache=True,
                eos_token_id=tokenizer.eos_token_id,
            )
            # out = model.language_model(outputs)
            out_ids = outputs.cpu().numpy()[0]

            if out_ids[-1] == tokenizer.eos_token_id:
                out_ids = out_ids[input_ids.shape[1]:-1]
            else:
                out_ids = out_ids[input_ids.shape[1]:]
            # print(f'first token: {first_token_id}')
            # print(f'context: {tokenizer.decode(inputs["context"])}')
            # print(f'question: {tokenizer.decode(inputs["question"])}')
            input_tokens = inputs["answer"]
            if input_tokens[-1] == tokenizer.eos_token_id:
                input_tokens = input_tokens[:-1]
            tqdm.write(f'Ans: {tokenizer.decode(out_ids)} Truth: {tokenizer.decode(input_tokens)}')
            output_ids.append(out_ids)

            if step % args.save_steps == 0:
                with open(args.output_path, 'wb') as file_out:
                    pickle.dump(output_ids, file_out)
        with open(args.output_path, 'wb') as file_out:
            pickle.dump(output_ids, file_out)