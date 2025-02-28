#!/bin/bash

MODEL_VERSION=vicuna-v1-5-7b
# gpu_vis=0 # per_device_train_batch_size * gradient_accumulation_steps * n_gpus = 128
MASTER_PORT=$(( ($RANDOM % 10000) + 20000 ))
output_dir=$1
gpu_vis=$2
other_args=$3

gradient_accumulation_steps=$((128 / (8 * gpu_vis)))

# deepspeed --include localhost:$gpu_vis --master_port $MASTER_PORT vtimellm/train/train_mem.py \
deepspeed --num_gpus ${gpu_vis} --master_port $MASTER_PORT vtimellm/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --lora_enable True \
    --model_name_or_path ./checkpoints/vicuna-7b-v1.5 \
    --version v1 \
    --data_path ../VTimeLLM_long/data/vtimellm_train/stage2.json \
    --feat_folder ../VTimeLLM_long/data/vtimellm_train/intern_clip_feat \
    --pretrain_mm_mlp_adapter ./checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin \
    --output_dir ${output_dir} \
    --bf16 True \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps ${gradient_accumulation_steps} \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 100 \
    --learning_rate 1e-4 \
    --freeze_mm_mlp_adapter True \
    --lora_r 64 \
    --lora_alpha 128 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to none \
    ${other_args} \

