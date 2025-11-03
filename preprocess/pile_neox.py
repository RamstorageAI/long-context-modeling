import os
from transformers import AutoTokenizer
import numpy as np
import zstandard as zstd
import subprocess
from glob import glob
from natsort import natsorted
from concurrent.futures import ThreadPoolExecutor
import multiprocessing
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
import shutil
from tqdm import tqdm
import pandas as pd


def extract_text_from_parquet(df):
    texts = []
    for index, row in df.iterrows():
        text = row['text']
        texts.append(text)
    return texts

def _tokenize_parquet(path, tokenizer, output_dir):
    texts = []
    lens = []

    filename = os.path.basename(path)
    base_name = filename 
    output_path = os.path.join(output_dir, base_name)
    output_path = output_path + ".data"

    
    if os.path.exists(output_path):
        print(f"File {output_path} already exists. Skipping...")
        return

    df = pd.read_parquet(path, engine='pyarrow')

    for index, row in df.iterrows():
        encoded = tokenizer.encode(row['text'])
        texts.extend(encoded)
        texts.append(tokenizer.eos_token_id)

    tokens = np.array(texts, dtype=np.uint16)
    tokens.tofile(output_path)
    out_size = os.path.getsize(output_path)
    print(f"tokenized: {output_path}, size: {out_size}")
        # with open(output_path + ".len.pkl", 'wb') as f:
        #     pickle.dump(lens, f)
    return
    

def process_file(file_path, input_root, output_root, tokenizer):
    """处理单个文件的原子操作"""
    # 构建相对路径
    rel_path = os.path.relpath(file_path, start=input_root)
    output_dir = os.path.join(output_root, os.path.dirname(rel_path))
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        _tokenize_parquet(file_path, tokenizer, output_dir)
    except Exception as e:
        print(f"\nError processing {os.path.basename(file_path)}: {str(e)}")
        raise

def concurrent_tokenize(input_path, output_path, tokenizer, max_workers=None):
    all_files = []
    all_files = [
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
        ]
    
    os.makedirs(output_path, exist_ok=True)
    
    workers = max_workers or multiprocessing.cpu_count()

    print(f"Processing {input_path} FILE-WISE concurrently with {workers} workers...")

    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = []
        for file_path in all_files:
            future = executor.submit(
                process_file,
                file_path,
                input_path,
                output_path,
                tokenizer
            )
            futures.append(future)
        
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"Critical error processing file: {str(e)}")


def concat_data_files_system(input_dir, output_file):
    chunk_dirs = []
    for root, dirs, _ in os.walk(input_dir):
        dirs[:] = natsorted(dirs, key=lambda x: x.lower())
        for d in dirs:
            if d.startswith("chunk"):
                chunk_dirs.append(os.path.join(root, d))

    chunk_dirs = natsorted(chunk_dirs, key=lambda x: x.lower())
    print(chunk_dirs)

    data_files = []
    for chunk in chunk_dirs:
        files = natsorted(
            glob(os.path.join(chunk, "*.data")),
            key=lambda x: os.path.basename(x).lower()
        )
        data_files.extend(files)

    if not data_files:
        raise FileNotFoundError(f"No .data files found in {input_dir}")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    open(output_file, 'wb').close()

    success_count = 0
    try:
        with open(output_file, 'ab') as f_out:
            # for idx, data_file in enumerate(data_files, 1):
            for idx, data_file in enumerate(tqdm(data_files, desc="Merging files", unit="file", ncols=80, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'), 1):
                # print(f"Merging [{idx}/{len(data_files)}] {os.path.basename(data_file)}")
                with open(data_file, 'rb') as f_in:
                    shutil.copyfileobj(f_in, f_out, 8 * 1024 * 1024)
                success_count += 1

        print(f"Success: Merged {success_count}/{len(data_files)} files -> {output_file}")

    except Exception as e:
        current_file = data_files[idx-1] if idx else "unknown"
        raise RuntimeError(
            f"Merge failed at file {current_file}: {e}"
        ) from e
    
if __name__ == "__main__":
    neox_tokenizer = AutoTokenizer.from_pretrained('EleutherAI/gpt-neox-20b')
    print (f"tokenizer.is_fast: {neox_tokenizer.is_fast}")

    concurrent_tokenize(
    input_path="EleutherAI/the_pile_deduplicated",
    output_path=" ",
    tokenizer=neox_tokenizer,
    max_workers=20
)

