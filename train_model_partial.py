from transformers import AutoModelForCausalLM, TrainingArguments, AutoConfig
from reader.lazy_loader import LazyChunkedLoader, LazyLoader
from reader.data_collator import LongRNNDataCollator
from reader.ruler_collator import RulerSynthesizer
from reader.dataset_new import TextDataset
import argparse
import os
import numpy as np
import torch
import re
from transformers import AutoTokenizer
from model.model_factory import create_model, load_pretrained
from transformers.modeling_utils import load_sharded_checkpoint
from peft import LoraConfig, get_peft_model


def _load_layer(mamba_dict, ramba_dict, mamba_idx, ramba_idx):
  pattern = rf'layers\.{mamba_idx}\.'
  for k, v in mamba_dict.items():
    if re.search(pattern, k):
      ramba_k = re.sub(pattern, f'layers.{ramba_idx}.', k)
      ramba_dict[ramba_k] = v
      # print(f"{ramba_k} loaded from {k}")


def load_mamba(model, state_dict, ramba_config, strict=True):
  # state_dict = torch.load(model_path, map_location=lambda a, b: a)
  mamba_dict = state_dict
  # for k, v in state_dict.items():
  #   new_k = k.replace('module.', '')
  #   mamba_dict[new_k] = v

  mamba_idx = 0
  ramba_idx = 0
  ramba_dict = {}

  for _ in range(ramba_config.num_lower_layers):
    _load_layer(mamba_dict, ramba_dict, mamba_idx, ramba_idx)
    mamba_idx += 1
    ramba_idx += 1

  if ramba_config.num_upper_groups > 0:
    # layer_types.append(LayerType.Encoder)
    ramba_idx += 1
    inner_groups = (ramba_config.num_upper_layers // ramba_config.num_upper_groups) // ramba_config.num_mamba_per_gca

    for _ in range(ramba_config.num_upper_groups):
      # layer_types.append(LayerType.Retrieval)
      ramba_idx += 1

      for _ in range(inner_groups):
        # layer_types.append(LayerType.GroupedCrossAttention)
        ramba_idx += 1
        for _ in range(ramba_config.num_mamba_per_gca):
          _load_layer(mamba_dict, ramba_dict, mamba_idx, ramba_idx)
          mamba_idx += 1
          ramba_idx += 1
    
  # ramba_dict['lm_head.weight']=mamba_dict['lm_head.weight']
  ramba_dict['backbone.embeddings.weight']=mamba_dict['backbone.embeddings.weight']
  ramba_dict['backbone.norm_f.weight']=mamba_dict['backbone.norm_f.weight']

  return model.load_state_dict(ramba_dict, strict=strict)

def show_trainable_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_ratio = trainable_params / total_params * 100
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Trainable ratio: {trainable_ratio:.2f}%")

def run(args):
    # model = Mamba2ForCausalLM(ramba_config)
    # model = create_model(args.model_type, args.config_path)
    print (args)
    if args.checkpoint_path is not None:
        if args.from_mamba:
            model = create_model(args.model_type, args.config_path)
            ramba_config = AutoConfig.from_pretrained(args.config_path)

            state_dict = {}
            base_dir = args.checkpoint_path
            idx = 1
            while True:
                shard_path = base_dir + f"/pytorch_model-0000{idx}-of-00003.bin"
                if os.path.exists(shard_path):
                    shard = torch.load(shard_path, map_location="cpu", weights_only=True)
                    state_dict.update(shard)
                    idx += 1
                    print(f'{shard_path} loaded')
                else:
                    break
            
            missing, unexpected = load_mamba(model, state_dict, ramba_config, strict=False)
            print (f"missing: {missing}")
            print (f"unexpected: {unexpected}")
        else:
            model = load_pretrained(args.model_type, 
                                    args.checkpoint_path, 
                                    config_path=args.config_path, 
                                    peft_adapter_path=args.peft_adapter_path,
                                    merge_and_unload=True)
    else:
        model = create_model(args.model_type, args.config_path)
    # print(model)
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    if 'ramba' in args.model_type:
        print(f'ramba models')
        modules_to_train = ["layers.32", "layers.33", "layers.34", "layers.43",  "layers.52", "layers.61"]
    else:
        print(f'mamba models')
        modules_to_train = None
    if args.stage == 1:
        print ("stage 1, load full Ramba model")
    
    elif args.stage == 2:
        print ("stage 2, load full Ramba model")
        lora_config = LoraConfig(
            r=128,
            target_modules=["in_proj", "out_proj"],
            # 5% of base_layer
            rank_pattern = {
                # "embeddings": 144,
                "in_proj": 128,
                "out_proj": 96
            },
            modules_to_save=modules_to_train,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM",
            use_rslora=True, 
        )

        model = get_peft_model(model, lora_config)
        # model.add_adapter(lora_config)
        # model.enable_adapters()
        # model.print_trainable_parameters()
        # for name, param in model.named_parameters():
        #     if param.requires_grad:
        #         print(name)
    elif args.stage == 3:
        print ("stage 3, fix Mamba tune HSA")
        modules_to_train=modules_to_train
        if modules_to_train is not None:
            for name, param in model.named_parameters():
                if any(layer in name for layer in modules_to_train):
                    param.requires_grad = True
                else:
                    param.requires_grad = False
    else:
        raise NotImplementedError

    rank = int(os.getenv("RANK", "0"))
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    print(f'local rank: {local_rank}')
    model.to(local_rank, non_blocking=True)
    # model.to('cuda', non_blocking=True)
    collator_fn = None
    if args.task_type == 'pretrain':
        data_collator = LongRNNDataCollator(
            pass_init_state=args.pass_init_state,
            neg_sampling_group=args.neg_sampling_group
        )
        collator_fn = data_collator.ramba_collator_fn
    elif args.task_type == 'sft_ruler':
        tokenizer = AutoTokenizer.from_pretrained(args.vocab_dir)
        data_collator = RulerSynthesizer(tokenizer, task_id=args.task_id, length=args.needle_len)
        collator_fn = data_collator.train_collate_fn
    else:
        raise Exception(f"Unsupported task type: {args.task_type}")

    # For Chunked train set
    if args.train_set_in_chunks:
        ds = LazyChunkedLoader(args.train_path, array_data_type=np.uint16)
    else:
        ds = LazyLoader(args.train_path, array_data_type=np.uint16)
    dataset = TextDataset(
        ds,
        batch_size=args.batch_size,
        segment_len=args.segment_len,
        num_samples = args.total_steps,
        ramdom_sampling=True,
        sample_across_doc=args.sample_across_doc,
        random_across_doc_sampling=args.random_across_doc_sampling,
        is_lazy=True,
        reset_state_samples=args.reset_state_samples
    )

    valid_ds = LazyLoader(args.valid_path, array_data_type=np.uint16)
    valid_dataset = TextDataset(
        valid_ds,
        batch_size=args.batch_size,
        segment_len=args.segment_len,
        num_samples = -1,
        ramdom_sampling=False,
        epochs=1,
        is_lazy=True
    )

    if args.model_type in ["mamba2", "mamba2_w_pass", "llama2-yarn"]:
        from trainer.mamba_trainer_wo_labels import MambaTrainer
    else:
        from trainer.mamba_trainer import MambaTrainer

    training_args = TrainingArguments(
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        num_train_epochs=1,
        eval_strategy="steps",
        metric_for_best_model="eval_loss",
        save_total_limit=4,
        label_names=["input_ids"],
        save_strategy=args.save_strategy,
        prediction_loss_only=True,
        dataloader_num_workers=args.num_workers,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim=args.optim,
        remove_unused_columns=False,
        warmup_ratio=args.warmup_ratio,
        output_dir=args.output_dir,
        logging_steps=args.log_steps,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        bf16=True,
        max_grad_norm=1.0,
        save_safetensors=False,
        ddp_find_unused_parameters=False,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={
            "min_lr_rate":args.min_lr_rate
        },
    )
    trainer = MambaTrainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=valid_dataset,
        data_collator=collator_fn,
        args=training_args,
    )
    trainer.train(resume_from_checkpoint=args.continue_training)
    

    if rank == 0:
        print(
            f' {rank} saving'
        )
        model.save_pretrained(args.output_dir, safe_serialization=False)
        print('save done')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=False, default=None)
    parser.add_argument("--checkpoint_path", type=str, required=False)
    parser.add_argument("--peft_adapter_path", type=str, required=False, default=None)
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--valid_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--warmup_ratio", type=float, default=0.02)
    parser.add_argument("--total_steps", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--segment_len", type=int, default=16384)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--min_lr_rate", type=float, default=0.2)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--enabling_ibs", action="store_true")
    parser.add_argument("--continue_training", action="store_true")
    # parser.add_argument("--reorg_prob", type=float, default=0.5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--log_steps", default=50, type=int)
    parser.add_argument("--eval_steps", default=1000, type=int)
    parser.add_argument("--train_set_in_chunks", action="store_true", help="train set in chunks")

    # parser.add_argument("--save_strategy", default='best', type=str)
    parser.add_argument("--pass_init_state", action="store_true")
    parser.add_argument("--sample_across_doc", action="store_true")
    parser.add_argument("--random_across_doc_sampling", action="store_true")

    parser.add_argument('--model_type', type=str, required=True, choices=['ramba_new', 'ramba_peft', "mamba2", "mamba_nsa"])
    parser.add_argument('--vocab_dir', type=str)
    # parser.add_argument("--safetensor", action="store_true")
    # parser.add_argument('--sharded', action='store_true')
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument('--reset_state_samples', default=-1, type=int)
    parser.add_argument('--neg_sampling_group', default=1, type=int)
    parser.add_argument('--task_type', type=str, default='pretrain', choices=['pretrain', 'sft_ruler'])
    parser.add_argument('--task_id', type=int, default=-1)
    parser.add_argument('--needle_len', type=int, default=7)
    parser.add_argument('--stage', type=int, required=True)
    parser.add_argument('--from_mamba', action='store_true')
    parser.add_argument('--save_strategy', type=str, default='best', choices=['best', 'steps'])
    args = parser.parse_args()

    run(args)


