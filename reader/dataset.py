from bisect import bisect_right
from itertools import accumulate
from torch.utils import data
import numpy as np
import random
import torch
import torch.distributed as dist


class LongRNNDataset(data.Dataset):
    
    def __init__(self, ds,
                 segment_len=1024,
                 segment_size=64,
                 batch_size=1,
                 epochs=10,
                 stride=-1,
                 num_samples=-1,
                 ramdom_sampling=True,
                 **kwargs):
        """
        sentence_start: the stripped article must start with a complete sentence
        """
        self.ds = ds
        self.segment_len = segment_len
        self.segment_size = segment_size
        self.batch_size = batch_size
        self.ramdom_sampling = ramdom_sampling
        if num_samples == -1:
            self.num_samples = (self.ds.total_tokens // segment_len) * epochs
        else:
            self.num_samples= num_samples

        self.weighting, self.total_len = None, None
        self.total_tokens = self.ds.total_tokens
        self.stride=stride
        if stride != -1:
            self.num_samples = (self.ds.total_tokens // stride) * epochs


    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # init rng
        if self.ramdom_sampling:
            if not dist.is_initialized():
                rng_idx = idx // self.segment_size
                rng = random.Random(rng_idx)
                rng = np.random.RandomState(seed=[rng.randint(0, 2 ** 32 - 1) for _ in range(16)])

                start = rng.randint(self.total_tokens - self.segment_len - 1)
                start += (idx % self.segment_size) * self.segment_len
                # start = min(start, self.ds.total_tokens - self.segment_len - 1)
                start = start % (self.ds.total_tokens - self.segment_len - 1)
                tokens = self.ds.file_read(start, start + self.segment_len)
                assert len(tokens) == self.segment_len
                return torch.tensor(np.array(tokens, dtype=np.int32)), idx % self.segment_size == 0
            else:
                world_size = dist.get_world_size()
                world_id = (idx // self.batch_size) % world_size
                batch_id = idx % self.batch_size
                assert world_id == dist.get_rank(), f'{world_id} != {dist.get_rank()}'
                samples_per_batch = self.num_samples // world_size // self.batch_size
                step = idx // world_size // self.batch_size
                trans_idx = world_id * self.batch_size * samples_per_batch + batch_id * samples_per_batch + step

                rng_idx = trans_idx // self.segment_size
                seg_bias = trans_idx % self.segment_size
                rng = random.Random(rng_idx)
                rng = np.random.RandomState(seed=[rng.randint(0, 2 ** 32 - 1) for _ in range(16)])

                start = rng.randint(self.total_tokens - self.segment_len - 1)
                start += seg_bias * self.segment_len
                start = start % (self.ds.total_tokens - self.segment_len - 1)
                # if dist.get_rank() == 0:
                #     print(f'world size: {world_size}, start {start}, rank: {dist.get_rank()}, {trans_idx}, rng_idx: {rng_idx}, seg_bias: {seg_bias}, reset: {seg_bias % self.segment_size == 0}')
                tokens = self.ds.file_read(start, start + self.segment_len)
                return torch.tensor(np.array(tokens, dtype=np.int32)), seg_bias % self.segment_size == 0
        else:
            start = idx * self.segment_len
            start = min(start, self.ds.total_tokens - self.segment_len - 1)
            tokens = self.ds.file_read(start, start + self.segment_len)
            return torch.tensor(np.array(tokens, dtype=np.int32)), idx % self.segment_size == 0



    def getidx(self, data_idx):
        token_ids = self.ds[data_idx]

        return token_ids