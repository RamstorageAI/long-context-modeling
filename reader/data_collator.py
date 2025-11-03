import torch
import random
import numpy as np


class LongRNNDataCollator:

    def __init__(self, pass_init_state=False, neg_sampling_group=1):
        self.pass_init_state = pass_init_state
        self.neg_sampling_group=neg_sampling_group
        print(f'enable pass init state: {self.pass_init_state}')

    def ramba_collator_fn(self, batch):
        '''
            input_list: [{"text": ..., "sentence_splits":...},...]
        '''
        # split sentence ids with negative ids
        chunk_ids_list = []
        # retrieval_attn_masks = np.zeros((N, max_input_len, 2 * N * max_input_len), dtype=bool)  # (N, L, 2NL)
        pass_state_ids = []
        neg_sampling = []
        for batch_id, (sent_tensor, pass_state) in enumerate(batch):
            # chunk_ids_list.append(sent_tensor)
            neg_sampling.append(sent_tensor)
            if len(neg_sampling) == self.neg_sampling_group or len(batch) == 1:
                cat_ids = torch.cat(neg_sampling, dim=0)  # (sampling_group * L)
                chunk_ids_list.append(cat_ids)
                neg_sampling = []
            
                if pass_state:
                    pass_state_ids.append(batch_id // self.neg_sampling_group)

        if self.pass_init_state:
            # padding
            # return {"input_ids": torch.stack(chunk_ids_list), "reset_init_state": reset_init_state}
            # return {"input_ids": torch.stack(chunk_ids_list), "reset_init_state": torch.tensor(reset_ids, dtype=torch.long)}
            # print(f'input ids shape: {torch.stack(chunk_ids_list).shape}, {pass_state_ids}')
            return {"input_ids": torch.stack(chunk_ids_list), "pass_init_state": torch.tensor(pass_state_ids, dtype=torch.long)}
        else:
            return {"input_ids": torch.stack(chunk_ids_list)}

class RNNTruncatedDataCollator:

    def __init__(self, segments=2, pass_state_prob=0.75):
        self.segments = segments
        self.pass_state_prob = pass_state_prob

    def ramba_collator_fn(self, batch):
        '''
            input_list: [{"text": ..., "sentence_splits":...},...]
        '''
        # split sentence ids with negative ids
        chunk_ids_list = []
        pass_state_ids = []

        # retrieval_attn_masks = np.zeros((N, max_input_len, 2 * N * max_input_len), dtype=bool)  # (N, L, 2NL)
        for batch_id, (sent_tensor, reset_init_state) in enumerate(batch):
            L = sent_tensor.shape[0]
            seg_len = L // self.segments
            for start in range(0, L, L // self.segments):
                chunk_ids_list.append(sent_tensor[start: start + L // self.segments])
                reorg = np.random.binomial(1, self.pass_state_prob, size=1)[0]
                if reorg == 1:
                    pass_state_ids.append(batch_id * self.segments + start // seg_len)

        # print(pass_state_ids)
        # padding
        # return {"input_ids": torch.stack(chunk_ids_list), "reset_init_state": reset_init_state}
        # return {"input_ids": torch.stack(chunk_ids_list), "reset_init_state": torch.tensor(reset_ids, dtype=torch.long)}
        return {"input_ids": torch.stack(chunk_ids_list), "pass_init_state": torch.tensor(pass_state_ids, dtype=torch.long)}

class BOSLongRNNDataCollator:

    def __init__(self, bos_token_id=50256):
        self.bos_token_id = bos_token_id

    def ramba_collator_fn(self, batch):
        '''
            input_list: [{"text": ..., "sentence_splits":...},...]
        '''
        # split sentence ids with negative ids
        chunk_ids_list = []
        reset_ids = []
        batch_size = len(batch)

        # retrieval_attn_masks = np.zeros((N, max_input_len, 2 * N * max_input_len), dtype=bool)  # (N, L, 2NL)
        for batch_id, (sent_tensor, reset_init_state) in enumerate(batch):
            new_sent_tensor = torch.zeros_like(sent_tensor)
            new_sent_tensor[0] = self.bos_token_id
            new_sent_tensor[1:] = sent_tensor[:-1]
            chunk_ids_list.append(sent_tensor)
            reset_ids.append(batch_id)

        # padding
        # return {"input_ids": torch.stack(chunk_ids_list), "reset_init_state": reset_init_state}
        return {"input_ids": torch.stack(chunk_ids_list)}

class IBSDataCollator:

    def __init__(self, reorg_prob=0.5):
        self.reorg_prob = reorg_prob

    def ramba_collator_fn(self, batch):
        '''
            input_list: [{"text": ..., "sentence_splits":...},...]
        '''
        reorg = np.random.binomial(1, self.reorg_prob, size=1)[0]
        chunk_ids_list = []
        reset_ids = []
        if reorg == 1:
            batch_size = len(batch)
            seg_len = batch[0][0].shape[0] // batch_size
            for batch_i in range(batch_size):
                current_ids = []
                for batch_j in range(batch_size):
                    st = batch_i * seg_len
                    ed = st + seg_len
                    current_ids.append(batch[batch_j][0][st:ed])
                chunk_ids_list.append(torch.cat(current_ids, dim=0))
                reset_ids.append(batch_i)
        else:
            for batch_id, (sent_tensor, reset_init_state) in enumerate(batch):
                chunk_ids_list.append(sent_tensor)
                reset_ids.append(batch_id)

        return {"input_ids": torch.stack(chunk_ids_list)}


def insert_id_every_x_elements(arr, x, id_value):
    num_ids_to_insert = len(arr) // x
    
    new_length = len(arr) + num_ids_to_insert
    new_arr = np.empty(new_length, dtype=arr.dtype)
    
    new_arr[:new_length: x + 1] = id_value
    new_arr[np.arange(new_length) % (x + 1) != 0] = arr
    
    return new_arr

class PasskeyRetrievalDataCollator:
    
    def __init__(
        self, 
        tokenizer, 
        chunk_size=64, 
        vocab_low=0, 
        vocab_high=50256, 
        token_cnt=10, 
        chunk_retrieval=True, 
        segment_size=1
    ):
        self._pass_key_token_ids = tokenizer.encode('The passkey is:')
        # self._pass_key_token_ids = tokenizer.encode(' |One of the special magic numbers for long-context is:')
        self._wrapper_token_id = tokenizer.encode('|')
        self._pass_key_prompt_ids = tokenizer.encode('|What is the passkey? The passkey is')
        # self._pass_key_prompt_ids = tokenizer.encode(' What is the special magic number for long-context mentioned in the provided text? Answer: ')
        self._tokenizer = tokenizer
        self._chunk_size = chunk_size
        self._chunk_win_size = -1
        self._low = vocab_low
        self._high = vocab_high
        self._token_cnt = token_cnt
        self._chunk_retrieval = chunk_retrieval
        self._segments = segment_size
        print(f'token cnt: {token_cnt}, chunk_retrieval: {chunk_retrieval}, chunk_size: {chunk_size}')

    def update_chunk_size(self, chunk_size):
        self._chunk_win_size = chunk_size

    def fn(self, tensors_and_positions):
        '''
            input_list: [{"text": ..., "sentence_splits":...},...]
        '''
        # split sentence ids with negative ids
        chunk_ids_list = []
        labels = []
        final_poses = []

        total_len = len(tensors_and_positions[0][0])
        if self._chunk_retrieval:
            final_pos = total_len - (self._chunk_size - 1)
        else:
            final_pos = total_len - 1
        # retrieval_attn_masks = np.zeros((N, max_input_len, 2 * N * max_input_len), dtype=bool)  # (N, L, 2NL)
        pass_state_ids = []
        for group_i, tensor_and_position in enumerate(tensors_and_positions):
            token_ids, _ = tensor_and_position
            token_ids = token_ids.numpy()
            # randomly insert passkey from 0 to -128
            rng = random.Random(token_ids[0] * token_ids[-1])
            rng = np.random.RandomState(seed=[rng.randint(0, 2 ** 32 - 1) for _ in range(16)])
            total_len = len(token_ids)
            assert total_len % self._chunk_size == 0, f'total len: {total_len}, chunk_size: {self._chunk_size}'
            # passkey = chr(ord('a') + rng.randint(25))
            # passkey_id = self._tokenizer.encode(passkey)
            passkey_ids = np.array(rng.randint(self._low, self._high, size=self._token_cnt))
            passkey_ids_ = np.concatenate((self._wrapper_token_id, self._pass_key_token_ids, passkey_ids, self._wrapper_token_id))
            # print(passkey_ids_)
            # if self._chunk_win_size == -1:
            
            # TODO: change back
            # start = rng.randint(len(token_ids) - 2 * self._chunk_size - len(passkey_ids_))  # at least one chunk away
            start = rng.randint(len(token_ids) // self._segments - 2 * self._chunk_size - len(passkey_ids_))  # locate at the first sentence
            # print(start)
            # start = 0
            # start += len(token_ids) // self._segments

            # else:
            # start = rng.randint(len(token_ids) - (self._chunk_win_size + 1) * self._chunk_size, len(token_ids) - 2 * self._chunk_size - len(passkey_ids))
            # print(type(token_ids))
            new_array = np.insert(token_ids, start, passkey_ids_)
            prompt_ids = np.concatenate((self._pass_key_prompt_ids, passkey_ids))
            # print(prompt_ids)
            if self._chunk_retrieval:
                new_array = np.insert(new_array, total_len - self._chunk_size - len(prompt_ids) + 1 + len(passkey_ids), prompt_ids)
                new_array = new_array[:total_len]
                new_array[-(self._chunk_size - 1 - len(passkey_ids)):] = -100
                labels.append(new_array[total_len - (self._chunk_size - 1)])
            else:
                new_array = np.insert(new_array, total_len - len(prompt_ids), prompt_ids)
                new_array = new_array[:total_len]
                labels.append(new_array[final_pos])
            
            # mod_array = insert_id_every_x_elements(new_array, self._chunk_size, 91)
            # mod_array[mod_array < 0] = 0
            # print(f'mod_array: {self._tokenizer.decode(mod_array)}, final pos token: {self._tokenizer.decode(new_array[final_pos])}')
            # if not self._chunk_retrieval:
            #     print(new_array)
            #     print(self._tokenizer.decode(new_array))
            # if self._segments == 1:
            chunk_ids_list.append(torch.tensor(new_array, dtype=torch.long))
            
            final_poses = final_pos
            pass_state_ids.append(group_i)
            # else:
            #     seglen = total_len // self._segments
            #     correct_label = new_array[total_len - (self._chunk_size - 1)]
                
            #     for offset in range(0, total_len, seglen):
            #         chunk_ids_list.append(torch.tensor(new_array[offset: offset + seglen], dtype=torch.long))
            #         final_poses.append(-1)
            #     labels.append(correct_label)
            #     final_poses[-1] = final_pos
                
        return {"input_ids": torch.stack(chunk_ids_list), 
                "final_pos":final_poses, 
                "labels": torch.tensor(labels), 
                'pass_init_state': torch.tensor(pass_state_ids, dtype=torch.long)}


            
