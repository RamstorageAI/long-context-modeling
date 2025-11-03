import torch
import argparse
from transformers import EvalPrediction, AutoTokenizer
from reader.data_collator import LongRNNDataCollator
from model.ramba_config import RambaConfig
from transformers import AutoConfig, AutoTokenizer, TrainingArguments
# from model.modeling_mamba2 import Mamba2ForCausalLM
from model.modeling_mamba2_nsa import Mamba2ForCausalLM
from model.model_factory import load_pretrained
from reader.dataset_new import TextDataset
from reader.lazy_loader import LazyLoader, LazyChunkedLoader
from reader.ruler_collator import RulerSynthesizer
from trainer.mamba_trainer import MambaTrainer
import json
import numpy as np
from flash_attn.losses.cross_entropy import CrossEntropyLoss
import torch.distributed as dist


# def eval_accuracy(eval_pred: EvalPrediction):
#     pred, gold_labels = eval_pred.predictions, eval_pred.label_ids
#     acc = np.sum(pred == gold_labels) / pred.shape[0]
#     return {"eval_accuracy": acc}


def run(args):
        
    model = load_pretrained(args.model_type, args.checkpoint_path, args.config_path)
    ds = LazyChunkedLoader(args.train_path, array_data_type=np.uint16)
    dataset = TextDataset(
        ds,
        batch_size=args.batch_size,
        segment_len=args.segment_len,
        num_samples = args.total_steps,
        ramdom_sampling=True,
        is_lazy=True,
        reset_state_samples=args.reset_state_samples
    )

    tokenizer = AutoTokenizer.from_pretrained(args.vocab_dir)
    data_collator = RulerSynthesizer(tokenizer, task_id=args.task_id, length=args.needle_len)

    if args.model_type in ["original_mamba2", "mamba2_w_pass", "llama2-yarn"]:
        print('trainer wo labels')
        from trainer.mamba_trainer_wo_labels import MambaTrainer
    else:
        print('mamba trainer')
        from trainer.mamba_trainer import MambaTrainer

    trainer = MambaTrainer(
        model=model,
        train_dataset=dataset,
        data_collator=data_collator.train_collate_fn,
        args=TrainingArguments(
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            adam_beta1=args.adam_beta1,
            adam_beta2=args.adam_beta2,
            num_train_epochs=1,
            label_names=["labels"],
            save_strategy="steps",
            save_total_limit=1,
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
            bf16=True,
            save_safetensors=False,
            ddp_find_unused_parameters=False,
            lr_scheduler_type="cosine_with_min_lr",
            lr_scheduler_kwargs={
                "min_lr_rate":args.min_lr_rate
            }
        ),
     
    )

    # try:
    trainer.train(resume_from_checkpoint=args.continue_training)
    # except Exception as e:
    #     print(e)
    #     trainer.train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_path", type=str, required=False, default=None)
    parser.add_argument("--vocab_dir", type=str, required=True)
    parser.add_argument("--train_path", type=str, required=True)
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--warmup_ratio", type=float, default=0.02)
    parser.add_argument("--total_steps", type=int, required=True)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--segment_len", type=int, default=16384)
    parser.add_argument("--segment_size", type=int, default=2, help="cut a sample into s segments")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--min_lr_rate", type=float, default=0.2)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--weight_decay", type=float, default=0.001)
    parser.add_argument("--save_steps", type=int, default=1000)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--enabling_ibs", action="store_true")
    parser.add_argument("--continue_training", action="store_true")
    parser.add_argument("--reset_state_samples", type=int, default=64)
    parser.add_argument("--reorg_prob", type=float, default=0.5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--optim", type=str, default="adamw_torch")
    parser.add_argument("--log_steps", default=50, type=int)
    parser.add_argument("--eval_steps", default=1000, type=int)
    parser.add_argument('--task_id', default=-1, type=int)
    parser.add_argument('--needle_len', default=10, type=int)
    args = parser.parse_args()

    run(args)