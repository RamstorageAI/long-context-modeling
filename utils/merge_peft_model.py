import torch
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--peft_adapter_path", type=str, required=True)
    parser.add_argument("--model_type", type=str)
    parser.add_argument("--")
