
# torchrun --standalone --nnodes=1 --nproc_per_node=16 sft/sft_summarization.py \
#     --learning_rate 2e-3 \
#     --vocab_dir configs/gpt-neox-20b \
#     --min_lr_rate 0.2 \
#     --warmup_ratio 0.02 \
#     --epochs 5 \
#     --train_path /ossfs/workspace/antnlp/aaron.hx/corpus/raw_xsum_neox/train.pkl \
#     --max_len 4096 \
#     --max_sum_len 1024 \
#     --batch_size 8 \
#     --save_steps 2000 \
#     --num_workers 1 \
#     --eval_steps 500 \
#     --log_steps 20 \
#     --gradient_accumulation_steps 1 \
#     --checkpoint_path /ossfs/workspace/antnlp/aaron.hx/ramba_x1-4k-370M-w-pass-neg-sampling2/checkpoint-61440 \
#     --output_dir /ossfs/workspace/antnlp/aaron.hx/ramba_x1-4k-370M-w-pass-neg-sampling2_xsum \
#     --model_type ramba_new

torchrun --standalone --nnodes=1 --nproc_per_node=16 sft/sft_summarization.py \
    --learning_rate 2e-3 \
    --vocab_dir configs/gpt-neox-20b \
    --min_lr_rate 0.2 \
    --warmup_ratio 0.02 \
    --epochs 5 \
    --train_path /ossfs/workspace/antnlp/aaron.hx/corpus/raw_cnn/train.pkl \
    --max_len 4096 \
    --max_sum_len 1024 \
    --batch_size 8 \
    --save_steps 2000 \
    --num_workers 1 \
    --eval_steps 500 \
    --log_steps 20 \
    --gradient_accumulation_steps 1 \
    --continue_training \
    --checkpoint_path /ossfs/workspace/antnlp/aaron.hx/ramba_x1-4k-370M-w-pass-neg-sampling2/checkpoint-61440 \
    --output_dir /ossfs/workspace/antnlp/aaron.hx/ramba_x1-4k-370M-w-pass-neg-sampling2_cnn \
    --model_type ramba_new