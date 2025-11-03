# accelerate launch \
#     --config_file configs_accelerate/stage0.config \

# torchrun --standalone --nnodes=1 --nproc_per_node=8 train_model.py \
#     --config_path configs/ramba-370M/config.json \
#     --learning_rate 2e-3 \
#     --min_lr_rate 0.2 \
#     --warmup_ratio 0.02 \
#     --train_path /ossfs/workspace/antnlp/lengjiaqi.ljq/corpus/pile-deduplicated-tokenized \
#     --train_set_in_chunks \
#     --valid_path /ossfs/workspace/antnlp/aaron.hx/corpus/pg19_noex-20b/valid \
#     --total_steps 15728640 \
#     --segment_len 4096 \
#     --batch_size 8 \
#     --save_steps 2000 \
#     --num_workers 1 \
#     --eval_steps 500 \
#     --log_steps 20 \
#     --gradient_accumulation_steps 2 \
#     --output_dir /ossfs/workspace/antnlp/aaron.hx/ramba_x1-4k-370M-w-pass \
#     --model_type mamba2 \
#     --pass_init_state

# torchrun --standalone --nnodes=1 --nproc_per_node=1 train_model.py \
#     --config_path configs/ramba-370M/config_3hop.json \
#     --learning_rate 2e-3 \
#     --min_lr_rate 0.2 \
#     --warmup_ratio 0.02 \
#     --train_path  /ossfs/workspace/antnlp/lengjiaqi.ljq/corpus/pile-deduplicated-tokenized  \
#     --train_set_in_chunks \
#     --valid_path /ossfs/workspace/antnlp/aaron.hx/corpus/pg19_noex-20b/valid \
#     --total_steps 15728640 \
#     --segment_len 4096 \
#     --batch_size 8 \
#     --save_steps 2000 \
#     --num_workers 1 \
#     --eval_steps 500 \
#     --log_steps 20 \
#     --gradient_accumulation_steps 2 \
#     --output_dir /ossfs/workspace/antnlp/aaron.hx/ramba_x1-4k-370M-w-pass-hop3 \
#     --model_type ramba \
#     --pass_init_state

# torchrun --standalone --nnodes=1 --nproc_per_node=16 train_model.py \
#     --config_path configs/ramba-370M/config_enc1_4096_softmax.json \
#     --learning_rate 2e-3 \
#     --min_lr_rate 0.2 \
#     --warmup_ratio 0.02 \
#     --train_path  /mnt/antresearchnlp-p/common/data/pile-deduplicated-tokenized/ \
#     --train_set_in_chunks \
#     --valid_path /ossfs/workspace/antnlp/aaron.hx/corpus/pg19_noex-20b/valid \
#     --total_steps 7864320 \
#     --segment_len 8192 \
#     --batch_size 4 \
#     --save_steps 2000 \
#     --num_workers 1 \
#     --eval_steps 500 \
#     --log_steps 20 \
#     --gradient_accumulation_steps 2 \
#     --output_dir /ossfs/workspace/antnlp/aaron.hx/ramba_x1-4k-370M-w-pass-neg-sampling2 \
#     --model_type ramba \
#     --pass_init_state \
#     --reset_state_samples 64 \
#     --neg_sampling_group 2

# torchrun --standalone --nnodes=1 --nproc_per_node=16 train_model.py \
#     --config_path configs/ramba-370M/config_enc1_4096_softmax.json \
#     --learning_rate 2e-3 \
#     --min_lr_rate 0.2 \
#     --warmup_ratio 0.02 \
#     --train_path  /mnt/antresearchnlp-p/common/data/pile-deduplicated-tokenized/ \
#     --train_set_in_chunks \
#     --valid_path /ossfs/workspace/antnlp/aaron.hx/corpus/pg19_noex-20b/valid \
#     --total_steps 7864320 \
#     --segment_len 8192 \
#     --batch_size 4 \
#     --save_steps 2000 \
#     --num_workers 1 \
#     --eval_steps 500 \
#     --log_steps 20 \
#     --gradient_accumulation_steps 2 \
#     --output_dir /ossfs/workspace/antnlp/aaron.hx/ramba_x1-4k-370M-softmax-wseg \
#     --model_type ramba \
#     --pass_init_state \
#     --continue_training \
#     --reset_state_samples 64 \
#     --neg_sampling_group 2


# torchrun --standalone --nnodes=1 --nproc_per_node=16 train_model.py \
#     --config_path configs/ramba-370M/config_enc1_4096.json \
#     --learning_rate 2e-3 \
#     --min_lr_rate 0.2 \
#     --warmup_ratio 0.02 \
#     --train_path  /mnt/antresearchnlp-p/common/data/pile-deduplicated-tokenized/ \
#     --train_set_in_chunks \
#     --valid_path /ossfs/workspace/antnlp/aaron.hx/corpus/pg19_noex-20b/valid \
#     --total_steps 15728640 \
#     --segment_len 4096 \
#     --batch_size 8 \
#     --save_steps 2000 \
#     --num_workers 1 \
#     --eval_steps 500 \
#     --log_steps 20 \
#     --gradient_accumulation_steps 2 \
#     --output_dir /ossfs/workspace/antnlp/aaron.hx/ramba_x1-4k-370M-wo-pass \
#     --model_type ramba

# torchrun --standalone --nnodes=1 --nproc_per_node=16 train_model.py \
#     --config_path configs/ramba-370M/config_enc1_4096_softmax.json \
#     --learning_rate 2e-3 \
#     --min_lr_rate 0.2 \
#     --warmup_ratio 0.02 \
#     --train_path  /mnt/antresearchnlp-p/common/data/pile-deduplicated-tokenized/ \
#     --train_set_in_chunks \
#     --valid_path /ossfs/workspace/antnlp/aaron.hx/corpus/pg19_noex-20b/valid \
#     --total_steps 15728640 \
#     --segment_len 4096 \
#     --batch_size 8 \
#     --save_steps 2000 \
#     --num_workers 1 \
#     --eval_steps 500 \
#     --log_steps 20 \
#     --gradient_accumulation_steps 2 \
#     --output_dir /ossfs/workspace/antnlp/aaron.hx/ramba_x1-4k-370M-softmax-woseg \
#     --model_type ramba \
#     --pass_init_state \
#     --reset_state_samples 64

