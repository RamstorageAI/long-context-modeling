# from model.llama import LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers import AutoConfig, PreTrainedModel
from copy import deepcopy
import json
import torch as nn
import torch
from safetensors.torch import load_file
from transformers.modeling_utils import load_sharded_checkpoint


def create_model(model_type, config_path, lora_config_path=None) -> PreTrainedModel:
    config = AutoConfig.from_pretrained(config_path)
    if model_type == "mamba2":
        from model.modeling_mamba2 import Mamba2ForCausalLM

        model = Mamba2ForCausalLM(config=config)
        # model = Mamba2ForCausalLM.from_pretrained(config_path)
        print(
            "from transformers.models.mamba2.modeling_mamba2 import Mamba2ForCausalLM"
        )
    elif model_type == "ramba2_peft":
        from peft import LoraConfig, get_peft_model
        from model.modeling_mamba2 import Mamba2ForCausalLM

        lora_config = LoraConfig(
            r=128,
            target_modules=["in_proj", "out_proj"],
            rank_pattern={"in_proj": 128, "out_proj": 96},
            lora_alpha=16,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            use_rslora=True,
        )
        model = Mamba2ForCausalLM(config=config)
        model = get_peft_model(model, lora_config)
    # elif model_type == 'mamba2_ssm':
    #     from mamba_ssm.models.mixer_seq_simple import Mamba2
    elif model_type == "llama2-yarn":
        from model.modeling_llama_together_yarn import LlamaForCausalLM

        model = LlamaForCausalLM(config=config)
    elif model_type == "ramba":
        from model.modeling_ramba import Mamba2ForCausalLM

        model = Mamba2ForCausalLM(config=config)
    elif model_type == "mamba_nsa":
        from model.modeling_mamba2_nsa import Mamba2ForCausalLM

        model = Mamba2ForCausalLM(config=config)
    elif model_type == 'inf_attn':
        from model.modeling_qwen_transformers import Qwen2MoeForCausalLM
        model = Qwen2MoeForCausalLM(config=config)
    else:
        raise NotImplementedError(f"model_type: {model_type} is not implemented")
    return model


def load_pretrained(
    model_type, checkpoint_path, config_path=None, peft_adapter_path=None, merge_and_unload=False
):
    config = AutoConfig.from_pretrained(config_path) if config_path else None

    if model_type == "mamba2":
        # from model.modeling_mamba2 import Mamba2ForCausalLM
        from transformers.models.mamba2.modeling_mamba2 import Mamba2ForCausalLM

        model = Mamba2ForCausalLM.from_pretrained(checkpoint_path, config=config)
        print(
            "from transformers.models.mamba2.modeling_mamba2 import Mamba2ForCausalLM"
        )
        if peft_adapter_path is not None:
            raise Exception(f"Not not support peft adapter for model_type: {model_type}")
    elif model_type == 'mamba_peft':
        from peft import PeftModel
        from model.modeling_mamba2_nsa import Mamba2ForCausalLM

        if peft_adapter_path is None:
            raise ValueError("peft_adapter_path is required for mamba2_peft model")
        base_model = Mamba2ForCausalLM.from_pretrained(checkpoint_path, config=config)
        print(f'peft adapter path: {peft_adapter_path}')
        model = PeftModel.from_pretrained(base_model, peft_adapter_path)
        # model.load_adapter(peft_adapter_path)
        if merge_and_unload:
            model = model.merge_and_unload()
        print("Loading mamba2_peft model using PeftModel")
    elif model_type == "ramba_peft":
        from peft import PeftModel
        from model.modeling_ramba import Mamba2ForCausalLM

        if peft_adapter_path is None:
            raise ValueError("peft_adapter_path is required for ramba2_peft model")
        base_model = Mamba2ForCausalLM.from_pretrained(checkpoint_path, config=config)
        model = PeftModel.from_pretrained(base_model, peft_adapter_path)
        if merge_and_unload:
            model = model.merge_and_unload()
        print("Loading ramba2_peft model using PeftModel")

    elif model_type == "llama2-yarn":
        from model.modeling_llama_together_yarn import LlamaForCausalLM

        model = LlamaForCausalLM.from_pretrained(checkpoint_path, config=config)
        print("from model.modeling_llama_together_yarn import LlamaForCausalLM")

    elif model_type == "ramba":
        from model.modeling_ramba import Mamba2ForCausalLM

        print(f'start loading {model_type}')
        model = Mamba2ForCausalLM.from_pretrained(checkpoint_path, config=config)
        print("from model.modeling_ramba import Mamba2ForCausalLM")
        if peft_adapter_path is not None:
            raise Exception(f"Not not support peft adapter for model_type: {model_type}")
    elif model_type == "mamba_nsa":
        from model.modeling_mamba2_nsa import Mamba2ForCausalLM

        model = Mamba2ForCausalLM.from_pretrained(checkpoint_path, config=config)
        print("from model.modeling_mamba2_nsa import Mamba2ForCausalLM")
        if peft_adapter_path is not None:
            raise Exception(f"Not not support peft adapter for model_type: {model_type}")
    else:
        raise NotImplementedError(f"model_type: {model_type} is not implemented")

    return model


def load_checkpoint(model, checkpoint_path, sharded=False, safetensor=False):
    if not sharded:
        if safetensor:
            state_dict = load_file(checkpoint_path)
        else:
            state_dict = torch.load(checkpoint_path, weights_only=True)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
    else:
        missing, unexpected = state_dict = load_sharded_checkpoint(
            model, checkpoint_path
        )

    if missing or unexpected:
        print(f"Warning:\n missing: {missing}, unexpected: {unexpected}")
