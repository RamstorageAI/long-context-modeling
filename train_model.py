from transformers import AutoModelForCausalLM, TrainingArguments, AutoConfig
from reader.lazy_loader import LazyChunkedLoader, LazyLoader
from reader.data_collator import LongRNNDataCollator
from reader.dataset_new import TextDataset
import argparse
import numpy as np
from torch import nn
import torch
import re
from safetensors.torch import load_file
from pathlib import Path
from model.model_factory import create_model
from transformers.modeling_utils import load_sharded_checkpoint


def show_trainable_params(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    trainable_ratio = trainable_params / total_params * 100
    print(f"Total parameters: {total_params}")
    print(f"Trainable parameters: {trainable_params}")
    print(f"Trainable ratio: {trainable_ratio:.2f}%")

def run(args):
    # model = Mamba2ForCausalLM(ramba_config)
    model = create_model(args.model_type, args.config_path)
    model = model.to(torch.bfloat16)
    
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    data_collator = LongRNNDataCollator(
        pass_init_state=args.pass_init_state,
        neg_sampling_group=args.neg_sampling_group
    )

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

    if args.model_type in ["original_mamba2", "mamba2_w_pass", "llama2-yarn"]:
        from trainer.mamba_trainer_wo_labels import MambaTrainer
    else:
        from trainer.mamba_trainer import MambaTrainer



    trainer = MambaTrainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator.ramba_collator_fn,
        args=TrainingArguments(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            adam_beta1=args.adam_beta1,
            adam_beta2=args.adam_beta2,
            num_train_epochs=1,
            eval_strategy="steps",
            metric_for_best_model="eval_loss",
            save_total_limit=2,
            label_names=["input_ids"],
            save_strategy="steps",
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
            save_safetensors=False,
            ddp_find_unused_parameters=False,
            lr_scheduler_type="cosine_with_min_lr",
            lr_scheduler_kwargs={
                "min_lr_rate":args.min_lr_rate
            },
        ),
    )
    trainer.train(resume_from_checkpoint=args.continue_training)
    model.save_pretrained(args.output_dir, safe_serialization=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=True)
    # parser.add_argument("--checkpoint_path", type=str, required=True)
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

    parser.add_argument("--pass_init_state", action="store_true")
    parser.add_argument("--sample_across_doc", action="store_true")
    parser.add_argument("--random_across_doc_sampling", action="store_true")

    parser.add_argument('--model_type', type=str, required=True)
    # parser.add_argument("--safetensor", action="store_true")
    # parser.add_argument('--sharded', action='store_true')
    parser.add_argument('--gradient_checkpointing', action='store_true')
    parser.add_argument('--reset_state_samples', default=-1, type=int)
    parser.add_argument('--neg_sampling_group', default=1, type=int)
    args = parser.parse_args()

    run(args)


