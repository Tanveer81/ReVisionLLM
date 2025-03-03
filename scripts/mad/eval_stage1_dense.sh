#!/bin/bash -l
#SBATCH --job-name=eval_stage1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=%x.txt
# -C a100_80
echo "Running Split:" $1
RUN=/checkpoints/mad_stage1_dense/

srun python vtimellm/eval/eval_nlq_negative.py --split $1 --total_split $3 --batch $2 --clip_path checkpoints/clip/ViT-L-14.pt --model_base checkpoints/models--lmsys--vicuna-7b-v1.5/snapshots/3321f76e3f527bd14065daf69dad9344000a201d --pretrain_mm_mlp_adapter checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin --data_path data/mad/MAD_test.json --feat_folder data/mad/CLIP_L14_frames_features_5fps --stage2 $RUN --debug_window 125 --num_frames 250 --q_feat_dir data/mad/clip_L14_text_features --topk_pool True --mad_prompt mad_grounding --feature_fps 5 --adapter_input_dim 768 --score mean_entropy --score_merge add --normalize True --log_path $RUN
