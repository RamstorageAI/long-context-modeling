# torchrun --standalone --nnodes=1 --nproc_per_node=16 train_ramba_passkey.py \
#     --config_path configs/ramba-220M/config_sparse_gqa_2hop_24mamba_1enc_1lmk.json \
#     --learning_rate 2e-3 \
#     --vocab_dir configs/gpt2-small \
#     --min_lr_rate 0.2 \
#     --train_path ../../../antnlp/aaron.hx/corpus/pg19_gpt2/train \
#     --valid_path ../../../antnlp/aaron.hx/corpus/pg19_gpt2/valid \
#     --checkpoint_path ../../../antnlp/aaron.hx/ramba_220M_sparse_enc1_pg19_60k_hf_gqa2hop_24mamba_1enc_1lmk/checkpoint-60000/pytorch_model.bin \
#     --total_steps 192000 \
#     --segment_len 16384 \
#     --segment_size 1 \
#     --batch_size 2 \
#     --save_steps 20000 \
#     --num_workers 1 \
#     --eval_steps 100 \
#     --log_steps 50 \
#     --gradient_accumulation_steps 1 \
#     --output_dir ../../../antnlp/aaron.hx/ramba_220M_sparse_enc1_pg19_60k_hf_gqa2hop_24mamba_1enc_1lmk_passkey

# torchrun --standalone --nnodes=1 --nproc_per_node=16 train_ramba_passkey.py \
#     --config_path configs/ramba-220M/mamba_gca_lmk.json \
#     --learning_rate 2e-3 \
#     --vocab_dir configs/gpt2-small \
#     --min_lr_rate 0.2 \
#     --train_path ../../../antnlp/aaron.hx/corpus/pg19_gpt2/train \
#     --valid_path ../../../antnlp/aaron.hx/corpus/pg19_gpt2/valid \
#     --checkpoint_path ../../../antnlp/aaron.hx/ramba_220M_lmk_with_sa_1hop_pg19_trunc2_noneg_60k_hf/checkpoint-60000/pytorch_model.bin \
#     --total_steps 192000 \
#     --segment_len 16380 \
#     --segment_size 1 \
#     --batch_size 2 \
#     --save_steps 20000 \
#     --num_workers 1 \
#     --eval_steps 100 \
#     --log_steps 50 \
#     --gradient_accumulation_steps 1 \
#     --output_dir ../../../antnlp/aaron.hx/ramba_220M_lmk_with_sa_1hop_pg19_trunc2_noneg_60k_hf_passkey

