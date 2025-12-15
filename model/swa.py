import torch
import torch.nn as nn
from typing import Optional, Tuple, Union, Dict
from liger_kernel.transformers.rms_norm import LigerRMSNorm as RMSNorm
from liger_kernel.transformers.rope import liger_rotary_pos_emb
from flash_attn_interface import flash_attn_func


class SlidingWindowAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True

        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape
        self.sliding_window = config.sliding_window
        assert self.sliding_window is not None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: Tuple[torch.Tensor, torch.Tensor],
        past_key_value: Optional[Dict] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, -1, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        cos, sin = position_embeddings
        assert cos.shape[1] == query_states.shape[-2]

        query_states, key_states = liger_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_value is not None:

            past_key_cache, past_value_cache = past_key_value.get(self.layer_idx, (None, None))
            if past_key_cache is None and past_value_cache is None:
                past_key_cache = key_states
                past_value_cache = value_states
            else:
                key_states = torch.cat([past_key_cache, key_states], dim=-2)
                value_states = torch.cat([past_value_cache, value_states], dim=-2)

            # TODO: implement with torch.roll
            past_key_value[self.layer_idx] = (key_states[:, :, -self.sliding_window:, :].contiguous(), value_states[:, :, -self.sliding_window:, :].contiguous())

        if query_states.dtype not in (torch.float16, torch.bfloat16):
            query_states = query_states.to(torch.bfloat16)
            key_states = key_states.to(torch.bfloat16)
            value_states = value_states.to(torch.bfloat16)

        attn_output = flash_attn_func(
            query_states,
            key_states,
            value_states,
            softmax_scale=None,
            causal=True,
            window_size=(self.sliding_window, 0)  # diff with Llama
        )

        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output