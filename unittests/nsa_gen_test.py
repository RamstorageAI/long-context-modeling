from unittest import TestCase
import torch
from transformers import AutoConfig
import numpy as np

class TestGeneration(TestCase):
    def test_nsa_gen(self):
        from model.modeling_mamba2_nsa import Mamba2ForCausalLM

        device = torch.device("cuda:0")
        config_path = './configs/ramba_hf/ramba_hf_config_mamba2_nsa_unittest.json'
        torch.manual_seed(2357)
        config = AutoConfig.from_pretrained(config_path)
        # dtype = torch.float16
        model = Mamba2ForCausalLM(config)
        model.to(device)
        # model.to(torch.bfloat16)
        model.eval()

        chunk_size = config.chunk_size
        L = chunk_size * 2
        T = chunk_size * 4
        input_ids = torch.tensor(torch.randint(0, 100, (1, L)), device=device)
        print(input_ids.shape)
        with torch.amp.autocast('cuda', dtype=torch.bfloat16), torch.no_grad():
            outputs = model.generate(
                input_ids, 
                max_length=T, 
                cache_position=torch.zeros(1), 
                return_dict_in_generate=True, 
                output_scores=True
            )

        generated_ids = outputs.sequences
        with torch.amp.autocast('cuda', dtype=torch.bfloat16), torch.no_grad():
            ref_outputs = model(input_ids=torch.tensor(generated_ids, dtype=torch.long, device=device), use_cache=False)

        out_scores = torch.stack(outputs.scores, dim=1)  # (1, chunk_size, vocab_size)
        ref_scores = ref_outputs.logits  # (1, chunk_size * 5, vocab_size)
        print(f"Max diff: {(out_scores - ref_scores[:, -(T - L) - 1: -1]).abs().max()}")