from unittest import TestCase
import torch
from transformers import AutoConfig
from model.ramba_config import RambaConfig
from model.causal_retrieval import RetrievalLayer, ChunkKVManager, ChunkEncoder, GroupedCrossAttention


class TestCausalRetrievalModules(TestCase):
    # def testRetrieval(self):
    #     device = torch.device('cuda:0')
    #     config = AutoConfig.from_pretrained('configs/ramba-small/config.json')

    #     N = 4
    #     D = 16
    #     S = config.chunk_size + 1
    #     dim = config.hidden_size
    #     chunk_k = torch.tensor(torch.rand((N, D, S, dim), device=device))
    #     chunk_v = torch.tensor(torch.rand((N, D, S, dim), device=device))
    #     lmk_embs = torch.tensor(torch.rand((N, D, dim), device=device))
    #     chunk_mgr = ChunkKVManager()
    #     chunk_mgr.append(chunk_k, chunk_v, lmk_embs)


    #     layer = RetrievalLayer(config, group_idx=0, device=device)
    #     layer.to(device)
    #     x = torch.tensor(torch.rand((N, D * S, dim), device=device))
    #     result = layer(x, chunk_mgr)

    def testEncoderLayer(self):
        device = torch.device('cuda:0')
        # config = AutoConfig.from_pretrained('configs/ramba-small/config.json')
        config = RambaConfig(
            d_model = 768,
            num_lower_layers = 4,
            num_upper_layers = 4,
            num_upper_groups = 2,
            vocab_size = 50277,
            ssm_cfg = {"layer": "Mamba2"}
        )

        N = 8
        D = 64
        S = config.chunk_size + 1
        dim = config.d_model

        layer1 = ChunkEncoder(config, layer_idx=0, device=device)
        layer1.to(device)
        layer2 = RetrievalLayer(config, group_idx=0, device=device)
        layer2.to(device)
        layer3 = GroupedCrossAttention(config, layer_idx=0, group_idx=0, device=device)
        layer3.to(device)

        x = torch.tensor(torch.rand((N, D * S, dim), device=device))
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            x, chunk_k, chunk_v, weights, mem_k, mem_v, landmarks = \
                layer1(x, chunk_k=None, chunk_v=None, weights=None, mem_k=None, mem_v=None, landmarks=None)
            x, chunk_k, chunk_v, weights, mem_k, mem_v, landmarks = \
                layer2(x, chunk_k=chunk_k, chunk_v=chunk_v, weights=weights, mem_k=mem_k, mem_v=mem_v, landmarks=landmarks)
            x, chunk_k, chunk_v, weights, mem_k, mem_v, landmarks = \
                layer3(x, chunk_k=chunk_k, chunk_v=chunk_v, weights=weights, mem_k=mem_k, mem_v=mem_v, landmarks=landmarks)