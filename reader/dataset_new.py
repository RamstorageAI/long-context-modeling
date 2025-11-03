from bisect import bisect_right
from itertools import accumulate
from torch.utils import data
import numpy as np
import random
import torch
from reader.lazy_loader import LazyLoader, LazyChunkedLoader
import torch.distributed as dist



class TextDataset(data.Dataset):
    """
    Only support lazy loader for now.
    """
    def __init__(self, ds,
                 segment_len=1024,
                 num_samples=-1,
                 epochs=1,
                 weighted=True,
                 random_sampling=True,
                 sample_across_doc=True,
                 random_across_doc_sampling=True,
                 reset_state_samples=-1,
                 batch_size=1,
                 **kwargs):
        self.ds = ds
        self.ds_len = len(self.ds)
        self.num_samples = num_samples
        if num_samples == -1:   # use all the dataset, for evaluation
            self.num_samples = (self.ds.total_tokens // segment_len) * epochs
            assert isinstance(self.ds, LazyLoader), 'Only LazyLoader supports for non-random sampling.'
        else:
            self.num_samples= num_samples
        self.segment_len = segment_len
        self.weighted = weighted
        self.sample_across_doc = sample_across_doc
        self.random_across_doc_sampling = random_across_doc_sampling
        self.weighting, self.total_len = None, None
        self.is_lazy = True
        self.random_sampling = random_sampling
        self.reset_state_samples = reset_state_samples
        self.batch_size = batch_size
        # print ("Dataset length: " + str(len(self)))
        # if hasattr(self.ds, 'is_lazy') and self.ds.is_lazy:
        #     self.is_lazy = True
        self.init_weighting()
        # print (self.weighting)

    def init_weighting(self):
        if self.weighted:
            if self.is_lazy:
                lens = np.array([self.ds.get_text_len(idx) for idx in range(len(self.ds))])
            else:
                lens = np.array([len(d['text']) if isinstance(d, dict)
                                 else len(d) for d in self.ds])
            self.total_len = np.sum(lens)
            # print(f"Dataset document count {len(lens)}, token count {self.total_len}")
            self.weighting = list(accumulate(lens))
        else:
            self.weighting = None


    """
    Ramdomly select a document with length of each documents as weights
    """
    def get_weighted_samples(self, np_rng):
        if self.weighting is not None:
            idx = np_rng.randint(self.total_len)
            return bisect_right(self.weighting, idx)
            # return max(0, bisect_right(self.weighting, idx)-1)
        else:
            return np_rng.randint(self.ds_len)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # init rng
        rng = random.Random(idx)
        rng = np.random.RandomState(seed=[rng.randint(0, 2 ** 32 - 1) for _ in range(16)])

        # get length weighted random index from dataset
        doc_idx = self.get_weighted_samples(rng)
        doc_len = self.ds.get_text_len(doc_idx)
        tokens = None
        pass_state = (idx // self.batch_size // dist.get_world_size()) % self.reset_state_samples != 0 if self.reset_state_samples != -1 else True

        if self.random_sampling:
            if not self.sample_across_doc:
                tokens_to_skip = doc_len - self.segment_len
                doc_tokens = self.ds[doc_idx]        
                if tokens_to_skip >= 0:
                    token_start_idx = rng.randint(tokens_to_skip + 1)
                    tokens = doc_tokens[token_start_idx:token_start_idx+self.segment_len]
                    # for t in tokens:
                    #     assert t >= 0 and t < 50257
                else:
                    tokens = np.concatenate((doc_tokens, np.array([-100] * abs(tokens_to_skip), dtype=np.int32)))

                assert len(tokens) == self.segment_len
                return torch.tensor(np.array(tokens, dtype=np.int32)), pass_state
            else:
                if not self.random_across_doc_sampling:
                    assert self.segment_len < self.ds.total_tokens
                    start = rng.randint(self.ds.total_tokens - self.segment_len - 1)
                    assert start+self.segment_len <= self.ds.total_tokens
                    tokens = self.ds.file_read(start, start + self.segment_len)
                    assert len(tokens) == self.segment_len
                    return torch.tensor(np.array(tokens, dtype=np.int32)), pass_state
                else:
                    # randomly sample across doc
                    tokens = []
                    while (len(tokens) < self.segment_len):
                        remaining_tokens = self.segment_len - len(tokens)
                        assert remaining_tokens > 0
                        doc_idx = self.get_weighted_samples(rng)
                        doc_len = self.ds.get_text_len(doc_idx)
                        doc_tokens = self.ds[doc_idx]
                        start = rng.randint(doc_len)
                        end = min(start + remaining_tokens, doc_len)
                        tokens.extend(doc_tokens[start:end])
                        
                    assert len(tokens) == self.segment_len
                    return torch.tensor(np.array(tokens, dtype=np.int32)), pass_state

        else:
            assert isinstance(self.ds, LazyLoader), 'Only LazyLoader supports for non-random sampling.'
            start = idx * self.segment_len
            start = min(start, self.ds.total_tokens - self.segment_len - 1)
            tokens = self.ds.file_read(start, start + self.segment_len)
            return torch.tensor(np.array(tokens, dtype=np.int32)), pass_state
