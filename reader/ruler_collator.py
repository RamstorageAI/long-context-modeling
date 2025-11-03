import random
import numpy as np
from reader.dataset import LongRNNDataset
from reader.lazy_loader import LazyLoader
from torch.utils import data
from torch.utils.data import SequentialSampler
from tqdm import tqdm
import string
import torch
from scipy.special import zeta 
from enum import Enum



class RulerSynthesizer:
    def __init__(self, tokenizer, vocab_low = 100, task_id=-1, **kwargs):
        self.tokenizer = tokenizer
        self._low = vocab_low
        self._high = tokenizer.vocab_size
        self._eos_id = tokenizer.eos_token_id

        self._s_niah_needle_ids = \
            self.tokenizer.encode(' |One of the special magic numbers for long-context is:')
        self._s_niah_end = self.tokenizer.encode('|')
        self._s_niah_question = \
            self.tokenizer.encode(' What is the special magic number for long-context mentioned in the provided text? Answer: ')

        self._vt_question = \
            self.tokenizer.encode(' Find all variables that are assigned the value ')
        self._vt_question_answer = self.tokenizer.encode('. Answer: ')

        self._mq_template = '| One of the special magic numbers for {} is {}.|'
        self._mq_question = ' What are all the special magic numbers for {} mentioned in the provided text?'
        self._mq_answer = self.tokenizer.encode('. Answer: ')

        self._fwe_tempalte = "[INST] Read the following coded text and track the frequency of each coded word. Find the three most frequently appeared coded words. {context}\nQuestion: Do not provide any explanation. Please ignore the dots '....'. What are the three most frequently appeared words in the above coded text? [/INST] Answer: According to the coded text above, the three most frequently appeared words are:"
        # self._fwe_wo_prefix_tempalte = "{context}\nQuestion: Do not provide any explanation. Please ignore the dots '....'. What are the three most frequently appeared words in the above coded text? [/INST] Answer: According to the coded text above, the three most frequently appeared words are:"
        self.task_id = task_id
        self.kwargs = kwargs

    def generate_single_niah(self, inputs, length=7):
        # print(f'needle len: {length}')
        rng = random.Random(inputs[0] * inputs[-1])
        rng = np.random.RandomState(seed=[rng.randint(0, 2 ** 32 - 1) for _ in range(16)])
        # rand_val = random.randint(10**length, 10**(length+1) - 1)
        # passkey_ids = self.tokenizer.encode(f'{rand_val}')
        passkey_ids = np.array(rng.randint(self._low, self._high, size=length))

        passkey_ids_ = np.concatenate((self._s_niah_needle_ids, passkey_ids, self._s_niah_end))
        # print(passkey_ids_)
        # if self._chunk_win_size == -1:
        passkey_len = len(passkey_ids) + 1
        prompt_ids = np.concatenate((self._s_niah_question, passkey_ids, [self._eos_id]))
        
        total_len = len(inputs)
        start = rng.randint(len(inputs) - len(passkey_ids_) - len(prompt_ids))  # locate at the first sentence
        new_array = np.insert(inputs, start, passkey_ids_)
        new_array = np.insert(new_array, total_len - len(prompt_ids), prompt_ids)
        new_array = new_array[:total_len]

        return new_array, new_array[:-passkey_len], new_array[-passkey_len:]


    def _insert_needles_into_ids(self, input_ids, needles, rng):
        org_len = len(input_ids)
        numbers = list(range(0, org_len))
        rng.shuffle(numbers)
        indices = numbers[:len(needles)]
        for idx in range(len(needles)):
            input_ids = np.insert(input_ids, indices[idx], needles[idx])
            for j in range(idx + 1, len(needles)):
                if indices[j] > indices[idx]:
                    indices[j] += len(needles[idx])
        return input_ids


    def generate_variable_tracking(self, inputs, total_var=6, max_hops=2, varlen=7, **kwargs):
        rng = random.Random(inputs[0] * inputs[-1])
        rng = np.random.RandomState(seed=[rng.randint(0, 2 ** 32 - 1) for _ in range(16)])
        def generate_random_variable_name():
            letters = [rng.choice(list(string.ascii_uppercase)) for _ in range(5)]
            return ''.join(letters)

        answer_num = total_var // (max_hops + 1)
        answers = []
        while len(answers) < answer_num:
            rand_val = rng.randint(10**varlen, 10**(varlen+1) - 1)
            if f'{rand_val}' not in answers:
                answers.append(f'{rand_val}')

        
        var_names = []
        while len(var_names) < total_var:
            new_var_name = generate_random_variable_name()
            if new_var_name not in var_names:
                var_names.append(new_var_name)

        
        assignments = []
        for i in range(0, total_var, max_hops + 1):
            assignment1 = self.tokenizer.encode(f'|VAR {var_names[i]} = {answers[i // (max_hops + 1)]}|')
            assignment2 = self.tokenizer.encode(f'|VAR {var_names[i + 1]} = {var_names[i]}|')
            assignment3 = self.tokenizer.encode(f'|VAR {var_names[i + 2]} = {var_names[i + 1]}|')
            assignments.append(assignment1)
            assignments.append(assignment2)
            assignments.append(assignment3)

        needle_len = sum([len(ids) for ids in assignments])
        rng.shuffle(assignments)
        
        question_ids = self._vt_question + self.tokenizer.encode(answers[0])
        # answer_ids = self._vt_question_answer + self.tokenizer.encode(var_names[0]) + self.tokenizer
        answer = ', '.join(var_names[:max_hops + 1])
        _answer_ids = self.tokenizer.encode(answer)
        answer_ids = np.concatenate((self._vt_question_answer, _answer_ids, [self._eos_id]))
        input_ids_trunc = inputs[:-len(answer_ids) - len(question_ids)]
        input_ids_trunc = input_ids_trunc[:-needle_len]
        input_with_needle = self._insert_needles_into_ids(input_ids_trunc, assignments, rng)

        new_ids = np.concatenate((input_with_needle, question_ids, answer_ids))

        return new_ids, new_ids[:-len(_answer_ids) - 1], new_ids[-len(_answer_ids) - 1:]


    def generate_multi_query(self, inputs, total_var=6, var_name_len = 5, var_len=5, num_queries=2, **kwargs):
        rng = random.Random(inputs[0] * inputs[-1])
        rng = np.random.RandomState(seed=[rng.randint(0, 2 ** 32 - 1) for _ in range(16)])

        var_names = []
        var_vals = []
        while len(var_names) < total_var:
            var_ids = ''.join([rng.choice(list(string.ascii_uppercase)) for _ in range(var_name_len)])
            if var_ids not in var_names:
                var_names.append(var_ids)
        while len(var_vals) < total_var:
            rand_val = rng.randint(10**var_len, 10**(var_len+1) - 1)
            if f'{rand_val}' not in var_vals:
                var_vals.append(f'{rand_val}')

        needles = []
        for needle_i in range(total_var):
            needles.append(self.tokenizer.encode(self._mq_template.format(var_names[needle_i], var_vals[needle_i])))
    
        rng.shuffle(needles)
        question_vals = ' and '.join(var_names[:num_queries])
        question = self._mq_question.format(question_vals)
        question_ids = self.tokenizer.encode(question)

        answer = ' '.join(var_vals[:num_queries])
        _answer_ids = self.tokenizer.encode(answer)
        answer_ids = np.concatenate((self._mq_answer, _answer_ids, [self._eos_id]))

        needle_len = sum([len(ids) for ids in needles])
        
        input_ids_trunc = inputs[:-len(answer_ids) - len(question_ids)]
        input_ids_trunc = input_ids_trunc[:-needle_len]
        input_with_needle = self._insert_needles_into_ids(input_ids_trunc, needles, rng)

        new_ids = np.concatenate((input_with_needle, question_ids, answer_ids))

        return new_ids, new_ids[:-len(_answer_ids) - 1], new_ids[-len(_answer_ids) - 1:]

    def generate_frequent_words_extraction(self, inputs, var_len=5, alpha=2.0, vocab_size=2000, num_words=-1, incremental=10, **kwargs):
        # generate vocab
        rng = random.Random(inputs[0] * inputs[-1])
        rng = np.random.RandomState(seed=[rng.randint(0, 2 ** 32 - 1) for _ in range(16)])

        vocab = [''.join([rng.choice(list(string.ascii_lowercase)) for _ in range(var_len)]) for _ in range(vocab_size)]
        while len(set(vocab)) < vocab_size:
            vocab.append(''.join([rng.choice(list(string.ascii_lowercase)) for _ in range(var_len)]))
        vocab = sorted(list(set(vocab)))
        rng.shuffle(vocab)
        vocab[0] = '...' # treat the top ranked as noise

        # sample words
        template = self._fwe_tempalte
        def gen_text(num_words):
            k = np.arange(1, len(vocab)+1)
            sampled_cnt = num_words*(k**-alpha)/zeta(alpha)
            sampled_words = [[w] * zi for w, zi in zip(vocab, sampled_cnt.astype(int))]
            sampled_words = [x for wlst in sampled_words for x in wlst]
            rng.shuffle(sampled_words)
            return template.format(context=' '.join(sampled_words), query=''), vocab[1:4]
        
        max_len = len(inputs)
        if num_words > 0:
            num_words = num_words
            text, answer = gen_text(num_words)
            while len(self.tokenizer.encode(text)) > max_len:
                num_words -= incremental
                text, answer = gen_text(num_words)
        else:
            num_words = max_len // var_len # init
            text, answer = gen_text(num_words)
            while len(self.tokenizer.encode(text + ' '.join(answer))) < max_len - 1:
                # print(f"num_words: {num_words}, current_len: {len(self.tokenizer.encode(text + ' '.join(answer)))}")
                num_words = int(num_words * 1.1)
                text, answer = gen_text(num_words)
            # num_words -= incremental
            num_words = int(num_words / 1.1)
        text, answer = gen_text(num_words)
        new_ids = self.tokenizer.encode(text + ' '.join(answer)) + [self._eos_id]
        answer_len = len(self.tokenizer.encode(' '.join(answer))) + 1

        return new_ids, new_ids[:-answer_len], new_ids[-answer_len:]

    def single_token_eval_collate_fn(self, samples):
        chunk_ids_list = []
        ground_truth = []
        for _, (ids, _) in enumerate(samples):
            if self.task_id == 0:
                _, q, a = self.generate_single_niah(ids, **self.kwargs)
            elif self.task_id == 1:
                _, q, a = self.generate_multi_query(ids, **self.kwargs)
            elif self.task_id == 2:
                _, q, a = self.generate_variable_tracking(ids, **self.kwargs)
                last_idx = np.argwhere(a == 13).flatten()[-1]
                q = np.concatenate([q, a[:last_idx + 1]])
                a = a[last_idx + 1:]
            elif self.task_id == 3:
                _, q, a = self.generate_frequent_words_extraction(ids, **self.kwargs)
            
            chunk_ids_list.append(torch.tensor(np.concatenate([q, a])))
            ground_truth.append(torch.tensor(a))
            # chunk_ids_list.append(torch.tensor(q))
            # ground_truth.append(torch.tensor(a[0]))
            # print(self.tokenizer.decode(chunk_ids_list[-1]))
            # print('-' * 20)
            # print(self.tokenizer.decode(a))
            # print('~' * 20)
        return {"input_ids": torch.stack(chunk_ids_list), "labels": torch.stack(ground_truth)}


    def train_collate_fn(self, samples):
        chunk_ids_list = []
        pass_state_ids = []
        final_poses = []
        for group_i, (ids, pass_state) in enumerate(samples):
            if self.task_id == -1:
                task_id = random.randint(0, 3)
            else:
                task_id = self.task_id
            # print(f'task_id: {self.task_id}')
            if task_id == 0:
                new_ids, q, _ = self.generate_single_niah(ids, **self.kwargs)
            elif task_id == 1:
                new_ids, q, _ = self.generate_multi_query(ids, **self.kwargs)
            elif task_id == 2:
                new_ids, q, _ = self.generate_variable_tracking(ids, **self.kwargs)
            elif task_id == 3:
                new_ids, q, _ = self.generate_frequent_words_extraction(ids, **self.kwargs)
            final_poses.append(len(q))
            # print(self.tokenizer.decode(new_ids))
            # print('~' * 20)
            # print(self.tokenizer.decode(new_ids[final_poses[-1]:final_poses[-1] + 1])) 
            # print('*' * 20)
            if len(new_ids) < len(ids):
                pad_len = len(ids) - len(new_ids)
                new_ids = np.concatenate((new_ids, [-1] * pad_len))
            chunk_ids_list.append(torch.tensor(new_ids))
            if pass_state:
                pass_state_ids.append(group_i)
        # print(pass_state_ids)
        return {"input_ids": torch.stack(chunk_ids_list), 
                "final_pos":final_poses, 
                'pass_init_state': torch.tensor(pass_state_ids, dtype=torch.long)}


if __name__ == '__main__':
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained('configs/gpt2-small')
    max_seq_len = 512

    corpus_path = "../../../antnlp/aaron.hx/corpus/pg19_gpt2/train"
    valid_ds = LazyLoader(corpus_path)

    dataset = LongRNNDataset(
        valid_ds,
        batch_size=1,
        segment_len=max_seq_len,
        ramdom_sampling=False,
        segment_size=1,
        epochs=1
    )

    dataloader = data.DataLoader(dataset,
                                batch_size=1,
                                sampler=SequentialSampler(dataset),
                                num_workers=1
                                )

    synther = RulerSynthesizer(tokenizer)
    # ob = synther._insert_needles_into_ids(np.array([1,2,3,4,5,6,7]), [np.array([-1, -2]), np.array([-3, -4])])
    # print(ob)

    epoch_iterator = tqdm(dataloader, desc="Iteration")
    for idx, inputs in enumerate(epoch_iterator):
        # print(inputs)
        # print(tokenizer.decode(inputs[0][0].cpu().numpy()))
        ids = inputs[0][0].cpu().numpy()
        # new_ids, question, answer = synther.generate_single_niah(ids)
        new_ids, question, answer = synther.generate_variable_tracking(ids)
        
        # new_ids, question, answer = synther.generate_multi_query(ids)
        # new_ids, question, answer = synther.generate_frequent_words_extraction(ids)
        assert len(new_ids) <= len(ids), f'{len(new_ids)} > {len(ids)}'
        # print(f'len new ids: {len(new_ids)}')
        print(tokenizer.decode(new_ids))
        print('-' * 20)
        print(tokenizer.decode(question))
        print('*' * 20)
        last_idx = np.argwhere(answer == 11).flatten()[-1]
        print(tokenizer.decode(answer[last_idx + 1:-1]))
        print(tokenizer.decode(answer))
        print('~' * 20)
        
        if idx == 5:
            break