def convert_ssm_config_to_hf_config(config_ssm: Dict, mamba2_model_dict: Dict) -> Mamba2Config:
    """Convert a Mamba2Config from mamba_ssm to a Mamba2Config from here."""
    hf_config = Mamba2Config()

    # Switch to a different dict depending on model type
    config_dict = mamba2_model_dict

    # Set important values from config and recalculate other resulting entries
    hf_config.hidden_size = config_ssm[config_dict["hidden_size"]]
    hf_config.num_heads = (hf_config.hidden_size * hf_config.expand) // hf_config.head_dim
    hf_config.num_hidden_layers = config_ssm[config_dict["num_hidden_layers"]]
    hf_config.n_groups = config_ssm.get(config_dict["n_groups"], 1)
    hf_config.tie_word_embeddings = config_ssm["tie_embeddings"]
    hf_config.bos_token_id = config_dict["bos_token_id"]
    hf_config.pad_token_id = config_dict["pad_token_id"]
    hf_config.eos_token_id = config_dict["eos_token_id"]

    # Padded vocab size, mostly of 16 but 32 is also very common in different models
    vocab_size = config_ssm["vocab_size"]
    pad_vocab_size_multiple = config_ssm["pad_vocab_size_multiple"]
    if (vocab_size % pad_vocab_size_multiple) != 0:
        vocab_size += pad_vocab_size_multiple - (vocab_size % pad_vocab_size_multiple)
    hf_config.vocab_size = vocab_size

    return hf_config
