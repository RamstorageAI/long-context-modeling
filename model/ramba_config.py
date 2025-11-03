from dataclasses import dataclass, field, asdict
import json


@dataclass
class RambaConfig:

    d_model: int = 768
    d_intermediate: int = 0
    retrieval_dim: int = 384
    num_lower_layers: int = 6
    num_upper_layers: int = 6
    num_upper_groups: int = 2
    num_mamba_per_gca: int = 1
    encoder_layers: int = 2
    vocab_size: int = 50277
    ssm_cfg: dict = field(default_factory=dict)
    rms_norm: bool = True
    # Set fp32 to false
    residual_in_fp32: bool = False
    fused_add_norm: bool = True
    pad_vocab_size_multiple: int = 8
    tie_embeddings: bool = True
    sliding_window: int = 512
    chunk_size: int = 64
    chunk_topk: int = 8
    num_attention_heads: int = 12
    num_kv_heads: int = 6
    pad_id: int = 0
    lmk_encoder_layers: int = 1
    neg_sampling: bool = False
    insert_lmk: bool = False
    insert_mlp: bool = False
    lmk_id: int = 50276

    def to_json_string(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=4)