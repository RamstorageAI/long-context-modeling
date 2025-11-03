# torchrun --standalone --nnodes=1 --nproc_per_node=16 sft/sft_ruler.py \
#     --checkpoint_path /ossfs/workspace/antnlp/aaron.hx/ramba_x1-4k-370M-w-pass-neg-sampling2/checkpoint-61440 \
#     --learning_rate 2e-3 \
#     --vocab_dir configs/gpt-neox-20b \
#     --min_lr_rate 0.2 \
#     --train_path /mnt/antresearchnlp-p/common/data/pile-deduplicated-tokenized/ \
#     --total_steps 393216 \
#     --segment_len 4096 \
#     --batch_size 8 \
#     --model_type ramba_new \
#     --save_steps 1000 \
#     --num_workers 1 \
#     --log_steps 10 \
#     --gradient_accumulation_steps 1 \
#     --task_id 0 \
#     --output_dir ../../../antnlp/aaron.hx/ramba_x1-4k-370M-w-pass-neg-sampling2-rulersft-single
