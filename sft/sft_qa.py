import argparse

from transformers import AutoTokenizer, TrainingArguments
# from reader.dataset_xsum import SummarizationDataset, SummarizationCollator
from reader.dataset_qa import QADataset, QACollator
# from reader.lazy_loader import LazyLoader
# from reader.data_collator import LongRNNDataCollator, RNNTruncatedDataCollator
from trainer.mamba_trainer import MambaTrainer
import json
import torch
import numpy as np
from flash_attn.losses.cross_entropy import CrossEntropyLoss
from transformers import AutoConfig
# import torch.distributed as dist
from model.model_factory import load_pretrained, create_model


def run(args):
    model = load_pretrained(args.model_type, args.checkpoint_path, args.config_path)
    if args.model_type in ["original_mamba2", "mamba2_w_pass", "llama2-yarn"]:
        from trainer.mamba_trainer_wo_labels import MambaTrainer
    else:
        from trainer.mamba_trainer import MambaTrainer
    tokenizer = AutoTokenizer.from_pretrained(args.vocab_dir)
    dataset = QADataset(args.train_path, tokenizer, tokenizer.eos_token_id)
    data_collator = QACollator(args.max_len, args.max_sum_len, tokenizer=tokenizer, pad_id=tokenizer.eos_token_id, answer_loss_only=args.answer_loss_only)

    trainer = MambaTrainer(
        model=model,
        train_dataset=dataset,
        data_collator=data_collator.fn,
        args=TrainingArguments(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            adam_beta1=args.adam_beta1,
            adam_beta2=args.adam_beta2,
            num_train_epochs=args.epochs,
            label_names=["input_ids"],
            save_strategy="steps",
            save_total_limit=3,
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
            }
        ),
     
    )


    trainer.train(resume_from_checkpoint=args.continue_training)
    # model.save_pretrained(args.output_dir, safe_serialization=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=False, default=None)
    parser.add_argument("--vocab_dir", type=str, required=True)
    parser.add_argument('--epochs', type=int, required=True)
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--warmup_ratio", type=float, default=0.02)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_len", type=int, default=16384)
    parser.add_argument("--max_sum_len", type=int, default=4096)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--min_lr_rate", type=float, default=0.2)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--pass_state_prob", type=float, default=0.0)
    parser.add_argument("--enable_truncated_rnn", action="store_true")
    parser.add_argument('--pass_states', action='store_true')
    parser.add_argument("--trunc_segments", type=int, default=2)
    parser.add_argument("--continue_training", action="store_true")
    parser.add_argument("--reorg_prob", type=float, default=0.5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--log_steps", default=50, type=int)
    parser.add_argument("--eval_steps", default=1000, type=int)

    parser.add_argument("--answer_loss_only", action="store_true")
    args = parser.parse_args()

    run(args)