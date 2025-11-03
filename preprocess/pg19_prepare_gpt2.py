import os
from transformers import AutoTokenizer
import numpy as np
import pickle
import tiktoken


gpt2_tokenizer = AutoTokenizer.from_pretrained('your_tokenizer_dir')
gpt2_tokenizer = tiktoken.get_encoding("gpt2")
def _read_directory(path):
    texts = []
    lens = []
    for filename in os.listdir(path):
        if filename.endswith(".txt") and filename[:-4].isnumeric():
            print(filename)
            with open(os.path.join(path, filename), 'r') as f:
                # texts += gpt2_tokenizer.encode(f.read())
                encoded_texts = gpt2_tokenizer.encode(f.read())
                file_len = len(encoded_texts)
                texts.extend(encoded_texts)
                texts.append(gpt2_tokenizer.eot_token)
                lens.append(file_len+1)     # plus 1 for eos_token

    return np.array(texts, dtype=np.uint16), lens

src = "your-source-dir"
output_dir = "output-dir"

raw_eval_data, lens = _read_directory(src)
data_type = "data"
raw_eval_data.tofile(os.path.join(output_dir, data_type))
with open(os.path.join(output_dir, data_type + ".len.pkl"), 'wb') as f:
    pickle.dump(lens, f)
