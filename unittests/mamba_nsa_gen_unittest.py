from unittest import TestCase
import torch
from transformers import AutoConfig
from model.ramba_config import RambaConfig
import numpy as np

from model.causal_retrieval import RetrievalLayer, ChunkKVManager, ChunkEncoder, GroupedCrossAttention


class LogitsProcessorWithScore(torch.nn.Module):
    def __init__(self, prefix_allowed_token_fn):
        super().__init__()
        self.prefix_allowed_token_fn = prefix_allowed_token_fn
        self.scores = []

    def forward(self, input_ids, scores):
        # 保存原始 scores (logits)
        # print(scores.shape)
        self.scores.append(scores.clone())  # 保存当前 step 的分数

        # 根据prefix_allowed_token_fn限制 logits
        vocab_size = scores.size(-1)
        allowed_token_ids = self.prefix_allowed_token_fn(0, input_ids)
        mask = torch.full((vocab_size,), float("-inf")).to(scores.device)
        mask[list(allowed_token_ids)] = 0  # 只保留允许的 token
        scores += mask  # 更新 logits
        
        return scores

class TestGeneration(TestCase):
    def test_mamba_sw(self):
        from model.modeling_mamba2_nsa import Mamba2ForCausalLM
        # from model.modeling_mamba2 import Mamba2ForCausalLM
        device = torch.device("cuda:0")
        config_paths = [
            './configs/ramba_hf/ramba_hf_config_mamba2_sw512_unittest.json',
            './configs/ramba_hf/ramba_hf_config_mamba2_sw512_rope_unittest.json'
        ]
        torch.manual_seed(2357)
        for config_path in config_paths:
            # model = RambaLMHeadModel(config, device=device, dtype=dtype)
            config = AutoConfig.from_pretrained(config_path)
            # dtype = torch.float16
            model = Mamba2ForCausalLM(config)
            model.to(device)
            # model.to(torch.bfloat16)
            model.eval()
            dtype = torch.bfloat16

            chunk_size = config.chunk_size
            L = chunk_size * 2 + 2
            T = chunk_size * 4
            input_ids = torch.tensor(torch.randint(0, 100, (1, T)), device=device)
            # print(input_ids.shape)
            def force_given_tokens(batch_id, gen_ids):
                current_length = gen_ids.shape[1]
                # 如果生成长度小于给定的 tokens 长度，就强制生成给定的 token
                assert current_length < T
                return [input_ids[:, current_length]]
            logits_processor = LogitsProcessorWithScore(force_given_tokens)
            with torch.amp.autocast('cuda', dtype=dtype), torch.no_grad():
                outputs = model.generate(
                    input_ids[:, :L], 
                    max_length=T, 
                    cache_position=torch.zeros(1), 
                    return_dict_in_generate=True, 
                    logits_processor=[logits_processor],
                    output_scores=True,
                    eos_token_id=None,
                )

            generated_ids = outputs.sequences
            assert torch.all(generated_ids == input_ids)
            # input_ids = torch.tensor(torch.randint(0, 100, (1, T)), device=device)
            with torch.amp.autocast('cuda', dtype=dtype), torch.no_grad():
                ref_outputs = model(input_ids=input_ids, use_cache=False)

            out_scores = torch.stack(logits_processor.scores, dim=1)  # (1, chunk_size, vocab_size)
            print(f'out scores shape: {out_scores.shape}')
            ref_scores = ref_outputs.logits  # (1, chunk_size * 5, vocab_size)
            print(f"Max diff: {(out_scores - ref_scores[:, -(T - L) - 1: -1]).abs().mean()}")
            self.assertTrue((out_scores - ref_scores[:, -(T - L) - 1: -1]).abs().mean() < 0.005)
