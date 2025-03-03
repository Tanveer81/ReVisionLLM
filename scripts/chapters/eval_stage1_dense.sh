#!/bin/bash -l
#SBATCH --job-name=eval_stage1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=%x.txt
# -C a100_80
echo "Running Split:" $1
RUN=/checkpoints/chapters_stage1_dense/

srun python vtimellm/eval/eval_nlq_negative.py --split $1 --total_split $3 --batch $2 --clip_path checkpoints/clip/ViT-L-14.pt --model_base checkpoints/models--lmsys--vicuna-7b-v1.5/snapshots/3321f76e3f527bd14065daf69dad9344000a201d --data_path data/chapters/chapters_test.json --feat_folder data/chapters/chapters_clipl14_features/chapters_clipl14_features --q_feat_dir data/chapters/chapters_clip_L14_text_features --stage2 $RUN --log_path $RUN --debug_window 500 --num_frames 100 --adapter_input_dim 768 --feature_fps 2 --pretrain_mm_mlp_adapter checkpoints/vtimellm-vicuna-v1-5-7b-stage1/mm_projector.bin --load_ckp False --mad_prompt mad_grounding --debug False --topk_pool False --vis_feat_storage npy --score cosine_sim
