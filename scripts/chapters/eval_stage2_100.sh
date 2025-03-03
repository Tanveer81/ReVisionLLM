#!/bin/bash -l
#SBATCH --job-name=100_fix2
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a40:1
#SBATCH --output=/home/hpc/v100dd/v100dd11/sbatch/%x.txt

echo "Running Split:" $1

RUN=checkpoints/chapters_stage2_long_100


srun python vtimellm/eval/eval_nlq_retrieval_e2e2.py --clip_path checkpoints/clip/ViT-L-14.pt --model_base checkpoints/models--lmsys--vicuna-7b-v1.5/snapshots/3321f76e3f527bd14065daf69dad9344000a201d --pretrain_clip_adapter checkpoints/chapters_stage1_sparse/non_lora_trainables.bin     --pretrain_mm_mlp_adapter checkpoints/pretrain/mm_projector.bin \ --data_path data/chapters/chapters_test.json --feat_folder data/chapters/chapters_clipl14_features --stage2 $RUN --log_path $RUN/e2e2 --debug_window 500 --num_frames 250 --hierarchy_num_videos 100 --adapter_input_dim 768 --feature_fps 2 --mad_prompt mad_grounding --q_feat_dir data/chapters/chapters_clip_L14_text_features --hierarchy True --clip_adapter_text True --batch 100 --cross_attn True --clip_adapter_feature cls --hierarchy True --debug False --split $1 --total_split $3 --grounding_path /checkpoints/chapters_stage1_dense/predictions --vis_feat_storage npy