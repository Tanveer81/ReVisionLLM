#!/bin/bash -l
#SBATCH --job-name=eval_stage2_100
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:a100:1
#SBATCH --output=result-%x.txt
# -C a100_80

echo "Running Split:" $1


RUN=checkpoints/mad_stage2_long_100

srun python vtimellm/eval/eval_nlq_retrieval_e2e2.py --clip_path checkpoints/clip/ViT-L-14.pt --model_base checkpoints/models--lmsys--vicuna-7b-v1.5/snapshots/3321f76e3f527bd14065daf69dad9344000a201d --data_path data/mad/annotations/MAD-v1/MAD_test.json --feat_folder /home/atuin/v100dd/v100dd11/mad/ --stage2 $RUN --log_path $RUN/entropy --debug_window 125 --num_frames 250 --hierarchy_num_videos 100 --adapter_input_dim 768 --feature_fps 5 --mad_prompt mad_grounding --q_feat_dir data/mad/mad_data_for_cone/offline_lmdb/clip_L14_text_features --hierarchy True --clip_adapter True --clip_adapter_text True --batch 100

