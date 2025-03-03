#!/bin/bash

#SBATCH --job-name=chapters_stage1_sparse
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a40:8
#SBATCH --dependency=singleton   # job dependency
#SBATCH --open-mode=append
# -C a100_80
#SBATCH --output=/checkpoints/chapters_stage1_sparse/%x.txt


MODEL_VERSION=vicuna-v1-5-7b
gpu_vis=0 # per_device_train_batch_size * gradient_accumulation_steps * n_gpus = 128
MASTER_PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
CHECKPOINT=/nfs/data3/hannan/vtimellm/checkpoints
WINDOW=50
FRAMES=100
OUT_DIR=/checkpoints/chapters_stage1_sparse

ME=`basename "$0"`
echo "My slurm job id is $ME"

for dir in $OUT_DIR/checkpoint-*/ ; do
    echo "$dir"
    if [ ! -f $dir/zero_to_fp32.py ] || [ ! -f $dir/scheduler.pt ]; then  
        echo "File not found!" 
        rm -r $dir
    elif [ ! -f $dir/pytorch_model.bin ]; then
        echo "Creating pytorch model weight"
        python $dir/zero_to_fp32.py $dir $dir/pytorch_model.bin
    fi
done


deepspeed --master_port $MASTER_PORT vtimellm/train/train_mem.py \
    --deepspeed ./scripts/zero3.json \
    --lora_enable True \
    --training_stage 1 \
    --model_name_or_path checkpoints/models--lmsys--vicuna-7b-v1.5/snapshots/3321f76e3f527bd14065daf69dad9344000a201d \
    --stage2_path /checkpoints/mad_stage1_dense \
    --version v1 \
    --data_path data/chapters/chapters_train_partial_3.json \
    --feat_folder data/chapters/chapters_clipl14_features \
    --q_feat_dir data/chapters/chapters_clip_L14_text_features_train \
    --output_dir $OUT_DIR \
    --bf16 True \
    --num_train_epochs 1 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps .1 \
    --save_total_limit 1 \
    --learning_rate 1e-4 \
    --tune_mm_mlp_adapter True \
    --lora_r 64 \
    --lora_alpha 128 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 0 \
    --lazy_preprocess True \
    --report_to "tensorboard" \
    --num_frames $FRAMES \
    --debug_window $WINDOW \
    --dataset "mad" \
    --neg_window True \
    --neg_samples 1 \
    --neg_factor 1 \
    --feature_fps 5 \
    --adapter_input_dim 768 \
    --clip_adapter True \
    --clip_adapter_text True \
    --clip_adapter_feature cls \
    --stage1_load_lora True \
