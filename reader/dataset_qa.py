from bisect import bisect_right
from itertools import accumulate
from torch.utils import data
import pyarrow.parquet as pq
import numpy as np
import math
import random
import json
# from utils.misc import align_spans, get_sentence_from_words
from transformers import AutoTokenizer
from typing import Dict, List
import pickle
import torch.nn.functional as F
import torch
import os


def insert_id_every_x_elements(arr, x, id_value):
    num_ids_to_insert = len(arr) // x
    
    new_length = len(arr) + num_ids_to_insert
    new_arr = np.empty(new_length, dtype=arr.dtype)
    
    new_arr[:new_length: x + 1] = id_value
    new_arr[np.arange(new_length) % (x + 1) != 0] = arr
    
    return new_arr

class QACollator:
    def __init__(self, max_len, max_sum_len, chunk_size=-1, pad_id=0, tokenizer=None, answer_loss_only=False):
        self._tokenizer = tokenizer
        self._max_len = max_len
        self._max_sum_len = max_sum_len
        self._pad_id = pad_id
        self._chunk_size = chunk_size
        self._answer_loss_only = answer_loss_only

    def fn(self, batch):
        input_ids_list = []
        labels_list = []
        for item in batch:
            # prompt_ids = item['context'] + item['question']
            prompt_ids = np.concatenate([item['context'], item['question']])
            ans_ids = item['answer']
            # if self._chunk_size != -1:
            #     #pad input_ids to chunk_size
            #     total_len = math.ceil(prompt_ids.shape[0] / self._chunk_size) * self._chunk_size + 1
            #     # input_ids = F.pad(input_ids, (total_len - input_ids.shape[0], self._pad_id))
            #     assert type(self._pad_id) == int
            #     prompt_ids = np.concatenate(
            #         (np.array([self._pad_id] * (total_len - prompt_ids.shape[0]), dtype=np.int32), prompt_ids),
            #         dtype=np.int32
            #     )
            #     assert len(prompt_ids) % self._chunk_size == 1
            if len(prompt_ids) + len(ans_ids) > self._max_len:
                # keep sum_ids <= max_sum_len
                ans_ids = ans_ids[-self._max_sum_len: ]
                prompt_ids = prompt_ids[:(self._max_len - len(ans_ids))]
            assert len(prompt_ids) + len(ans_ids) <= self._max_len

            input_ids = np.concatenate((prompt_ids, ans_ids), dtype=np.int32)
            input_ids = np.concatenate((input_ids, np.array((self._max_len - len(input_ids)) * [-100])))
            
            if self._answer_loss_only:
                labels = np.full(self._max_len, -100, dtype=np.int32)
                labels[len(prompt_ids):len(prompt_ids) + len(ans_ids)] = ans_ids
            else:
                labels = input_ids

            input_ids_list.append(input_ids)
            # labels_list.append(np.where(labels == self._pad_id, -100, labels))
            labels_list.append(labels)

        # print (input_ids_list)
        # return {"input_ids": torch.tensor(input_ids_list, dtype=torch.long), "labels": torch.tensor(labels_list, dtype=torch.long)}
        input_ids_np = np.array(input_ids_list, dtype=np.int32)
        labels_np = np.array(labels_list, dtype=np.int32)

        return {
            "input_ids": torch.tensor(input_ids_np, dtype=torch.long),
            # "labels": torch.tensor(labels_np, dtype=torch.long),
        }

class QADataset(data.Dataset):
    
    def __init__(self, data_dir, tokenizer, eos_id, **kwargs):
        # data_name = data_dir + '/' + mode + '.json'
        self._tokenizer = tokenizer
        self._eos_id = eos_id
        self._lines = self.load_pickle(data_dir)

    def _to_ids(self, text):
        ids = self._tokenizer.encode(text)
        return ids

    def load_pickle(self, path):
        with open(path, 'rb') as file:
            input_items = pickle.load(file)

        return input_items

    def __getitem__(self, idx):
        return self._lines[idx]

    def __len__(self):
        return len(self._lines)

if __name__ == "__main__":
    # tokenizer = AutoTokenizer.from_pretrained("/ossfs/workspace/nas2/jipy/warpper/Generative-R2D2/data/newgpt2")
    # import tiktoken
    # tokenizer = tiktoken.get_encoding("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("/ossfs/workspace/antnlp/lengjiaqi.ljq/EleutherAI/gpt-neox-20b")
    xsumdataset = QADataset(data_dir="/ossfs/workspace/antnlp/lengjiaqi.ljq/squad/train.bin", tokenizer=tokenizer, eos_id=0)
    lendict = {"0:200":0,"200:500":0,"500:1024":0,"1024:2000":0, '2000:3000': 0, '>3000': 0}
    maxl = -1
    totall = 0
    for item in xsumdataset:
        length = len(item['answer'])
        if 0 <= length < 200:
            lendict["0:200"] += 1
        elif 200 <= length < 500:
            lendict["200:500"] += 1
        elif 500 <= length < 1024:
            lendict["500:1024"] += 1
        elif 1024 <= length < 2000:
            lendict["1024:2000"] += 1
        elif 2000 <= length < 3000:
            lendict["2000:3000"] += 1
        else:
            lendict[">3000"] += 1
        if length > maxl:
            maxl = length
        totall += length
    meanl = totall/len(xsumdataset)
    print(maxl, meanl, lendict)
    # print(len(xsumdataset))
    # text1 = tokenizer.convert_ids_to_tokens(xsumdataset[0]["text"])
    # text2 = tokenizer.convert_ids_to_tokens(xsumdataset[1]["text"])
    # print(xsumdataset[0], text1)
    # print("---------------------------------------next--------------------------------------------------")
    # print(xsumdataset[1], text2)
    # for i in range(1001):
    #     if len(tokenizer.encode(xsumdataset[i]["text"])) < 40:
    #         print("find!")
    #         print(len(tokenizer.convert_ids_to_tokens(xsumdataset[i]["text"])))
    #         print(xsumdataset[i], tokenizer.convert_ids_to_tokens(xsumdataset[i]["text"]))
    #         break
    # print("not find!")

# train: 35243 498.46054985613944 {'0:200': 35236, '200:500': 93042, '500:1000': 56554, '>1000': 19185}
# test: 13584 473.9429983234801 {'0:200': 2330, '200:500': 5102, '500:1000': 2939, '>1000': 962}
# valid: 6494 465.09075659927606 {'0:200': 2364, '200:500': 5164, '500:1000': 2852, '>1000': 947}