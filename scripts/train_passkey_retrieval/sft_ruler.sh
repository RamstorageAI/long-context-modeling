# torchrun --standalone --nnodes=1 --nproc_per_node=16 sft/sft_ruler.py \
#     --checkpoint_path /ossfs/workspace/antnlp/lengjiaqi.ljq/mb2-370M-wo-pass/checkpoint-61440 \
#     --learning_rate 2e-3 \
#     --vocab_dir configs/gpt-neox-20b \
#     --min_lr_rate 0.2 \
#     --train_path /mnt/antresearchnlp-p/common/data/pile-deduplicated-tokenized/ \
#     --total_steps 393216 \
#     --segment_len 4096 \
#     --batch_size 8 \
#     --model_type mamba_nsa \
#     --save_steps 1000 \
#     --num_workers 1 \
#     --log_steps 10 \
#     --gradient_accumulation_steps 1 \
#     --output_dir ../../../antnlp/aaron.hx/mamba2-370M-rulersft-vt

# torchrun --standalone --nnodes=1 --nproc_per_node=16 sft/sft_ruler.py \
#     --config_path /ossfs/workspace/nas2/aaron.hx/RAMLLM/configs/ramba-370M/config_sw_512_rope.json \
#     --checkpoint_path /ossfs/workspace/antnlp/aaron.hx/mamba_4k-370M-sw512-rope-w-pass/checkpoint-61440 \
#     --learning_rate 2e-3 \
#     --vocab_dir configs/gpt-neox-20b \
#     --min_lr_rate 0.2 \
#     --train_path /mnt/antresearchnlp-p/common/data/pile-deduplicated-tokenized/ \
#     --total_steps 393216 \
#     --segment_len 4096 \
#     --batch_size 8 \
#     --model_type mamba_nsa \
#     --save_steps 1000 \
#     --num_workers 1 \
#     --log_steps 10 \
#     --gradient_accumulation_steps 1 \
#     --output_dir ../../../antnlp/aaron.hx/mamba_rope-w-pass-neg-sampling2-rulersft-vt
