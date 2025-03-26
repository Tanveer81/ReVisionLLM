#!/bin/bash

#SBATCH --job-name=stage2_long_100
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a40:8
#SBATCH --dependency=singleton   # job dependency
#SBATCH --open-mode=append
# -C a100_80
#SBATCH --output=/home/hpc/v100dd/path/sbatch2/%x.txt


MODEL_VERSION=vicuna-v1-5-7b
gpu_vis=0 # per_device_train_batch_size * gradient_accumulation_steps * n_gpus = 128
MASTER_PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
WINDOW=125
FRAMES=250
OUT_DIR=checkpoints/mad_stage2_long_100

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

    # --pretrain_mm_mlp_adapter checkpoints/revisionllm-vicuna-v1-5-7b-stage1/mm_projector.bin \

deepspeed --master_port $MASTER_PORT revisionllm/train/train_mem.py \
    --deepspeed ./scripts/zero3-alternate.json \
    --lora_enable True \
    --training_stage 4 \
    --model_name_or_path checkpoints/models--lmsys--vicuna-7b-v1.5/snapshots/3321f76e3f527bd14065daf69dad9344000a201d  \
    --pretrain_mm_mlp_adapter checkpoints/stage1_sparse/non_lora_trainables.bin \
    --stage2_path checkpoints/mad_stage1_dense \
    --version v1 \
    --data_path data/mad/mad_stage2_fixed_prompt.json \
    --feat_folder data/mad/CLIP_L14_frames_features_5fps \
    --q_feat_dir data/mad/clip_L14_text_features \
    --output_dir $OUT_DIR \
    --bf16 True \
    --num_train_epochs 2 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps .1 \
    --save_total_limit 1 \
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
    --hierarchy True \
    --hierarchy_num_videos 100


    # --stage1_load_lora True \
    # --ignore_temporal True \


