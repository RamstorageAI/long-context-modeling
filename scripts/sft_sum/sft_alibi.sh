export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15


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
#     --checkpoint_path /ossfs/workspace/antnlp/aaron.hx/mamba_4k-370M-sw512-w-pass/checkpoint-61440 \
#     --config_path /ossfs/workspace/antnlp/aaron.hx/mamba_4k-370M-sw512-w-pass/checkpoint-61440/config.json \
#     --output_dir /ossfs/workspace/antnlp/lengjiaqi.ljq/mb_alibi_xsum \
#     --model_type mamba_nsa 

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
    --checkpoint_path /ossfs/workspace/antnlp/aaron.hx/mamba_4k-370M-sw512-w-pass/checkpoint-61440 \
    --config_path /ossfs/workspace/antnlp/aaron.hx/mamba_4k-370M-sw512-w-pass/checkpoint-61440/config.json \
    --output_dir /ossfs/workspace/antnlp/lengjiaqi.ljq/mb_alibi_cnn \
    --model_type mamba_nsa \
    --continue_training

torchrun --standalone --nnodes=1 --nproc_per_node=16 sft/sft_qa.py \
    --learning_rate 2e-3 \
    --vocab_dir configs/gpt-neox-20b \
    --min_lr_rate 0.2 \
    --warmup_ratio 0.02 \
    --epochs 10 \
    --train_path /ossfs/workspace/antnlp/lengjiaqi.ljq/ehovy/race/train.bin \
    --max_len 1024 \
    --max_sum_len 32 \
    --batch_size 32 \
    --save_steps 600 \
    --num_workers 1 \
    --eval_steps 500 \
    --log_steps 5 \
    --gradient_accumulation_steps 1 \
    --checkpoint_path /ossfs/workspace/antnlp/aaron.hx/mamba_4k-370M-sw512-w-pass/checkpoint-61440 \
    --config_path /ossfs/workspace/antnlp/aaron.hx/mamba_4k-370M-sw512-w-pass/checkpoint-61440/config.json \
    --output_dir /ossfs/workspace/antnlp/lengjiaqi.ljq/mb_alibi_race \
    --model_type mamba_nsa 

torchrun --standalone --nnodes=1 --nproc_per_node=16 sft/sft_qa.py \
    --learning_rate 2e-3 \
    --vocab_dir configs/gpt-neox-20b \
    --config_path configs/ramba-370M/config_nsa_1024.json \
    --min_lr_rate 0.2 \
    --warmup_ratio 0.02 \
    --epochs 10 \
    --train_path /ossfs/workspace/antnlp/lengjiaqi.ljq/squad/train.bin \
    --max_len 1024 \
    --max_sum_len 32 \
    --batch_size 32 \
    --save_steps 600 \
    --num_workers 1 \
    --eval_steps 500 \
    --gradient_accumulation_steps 1 \
    --checkpoint_path /ossfs/workspace/antnlp/aaron.hx/mamba_4k-370M-sw512-w-pass/checkpoint-61440 \
    --config_path /ossfs/workspace/antnlp/aaron.hx/mamba_4k-370M-sw512-w-pass/checkpoint-61440/config.json \
    --output_dir /ossfs/workspace/antnlp/lengjiaqi.ljq/mb_alibi_squad \
    --model_type mamba_nsa

torchrun --standalone --nnodes=1 --nproc_per_node=16 sft/sft_qa.py \
    --learning_rate 2e-3 \
    --vocab_dir configs/gpt-neox-20b \
    --config_path configs/ramba-370M/config_nsa_4096.json \
    --min_lr_rate 0.2 \
    --warmup_ratio 0.02 \
    --epochs 6 \
    --train_path /ossfs/workspace/antnlp/lengjiaqi.ljq/hotpot_qa/fullwiki/train.bin \
    --max_len 4096 \
    --max_sum_len 32 \
    --batch_size 8 \
    --save_steps 600 \
    --num_workers 1 \
    --eval_steps 600 \
    --log_steps 5 \
    --gradient_accumulation_steps 1 \
    --checkpoint_path /ossfs/workspace/antnlp/aaron.hx/mamba_4k-370M-sw512-w-pass/checkpoint-61440 \
    --config_path /ossfs/workspace/antnlp/aaron.hx/mamba_4k-370M-sw512-w-pass/checkpoint-61440/config.json \
    --output_dir /ossfs/workspace/antnlp/lengjiaqi.ljq/mb_alibi_hotpot \
    --model_type mamba_nsa
