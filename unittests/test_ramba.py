from unittest import TestCase
from model.ramba_mixer_model import RambaLMHeadModel
from model.ramba_config import RambaConfig
import torch


class TestRamba(TestCase):
    def testRamba(self):
        config = RambaConfig(
            d_model = 768,
            num_lower_layers = 4,
            num_upper_layers = 4,
            num_upper_groups = 2,
            vocab_size = 50277,
            ssm_cfg = {"layer": "Mamba2"}
        )
        device = torch.device('cuda:0')
        dtype = torch.bfloat16
        model = RambaLMHeadModel(config, device=device, dtype=dtype)
        model.to(device)
        
        input_ids = torch.tensor(torch.zeros(8, 63 * 16), dtype=torch.long, device=device)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            model(input_ids)