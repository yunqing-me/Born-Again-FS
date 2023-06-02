#!/usr/bin/env bash

export PYTHONPATH=.






# CUDA_VISIBLE_DEVICES=0 python train_linear.py --method protonet --dataset miniImagenet --testset cub --n_shot 1 --stop_epoch 400 \
#  --warmup baseline --train_aug --model ResNet10 --name ban_episodic_gen_0 \
#  --wandb --wandb_project_name ban_episodic_gen_0 --wandb_run_name run1 \
# & \

# CUDA_VISIBLE_DEVICES=1 python train_linear.py --method protonet --dataset miniImagenet --testset cub --n_shot 1 --stop_epoch 400 \
#  --warmup baseline --train_aug --model ResNet10 --name ban_episodic_gen_0 \
#  --wandb --wandb_project_name ban_episodic_gen_0 --wandb_run_name run1 \
# & \


# CUDA_VISIBLE_DEVICES=2 python train_baseline_distill_gen_1_single.py --method baseline --dataset miniImagenet --gen 1 --stop_epoch 400 \
#  --warmup baseline --train_aug --model ResNet10 --name baseline_gen_1 \

# CUDA_VISIBLE_DEVICES=1 python train_proto_gen_1.py --method protonet --dataset miniImagenet --testset cub --n_shot 1 --stop_epoch 500 \
#  --warmup baseline --train_aug --model ResNet10 --name ban_episodic_gen_1 \
#  --wandb --wandb_project_name ban_episodic_gen_1 --wandb_run_name run1 \


# CUDA_VISIBLE_DEVICES=4 python train_baseline_distill_gen_1_single.py --method baseline --dataset miniImagenet --gen 2 --stop_epoch 400 \
#  --warmup baseline --train_aug --model ResNet10 --name baseline_gen_2 \
# & \
CUDA_VISIBLE_DEVICES=1 python train_proto_gen_1.py --method protonet --dataset miniImagenet --testset cub --n_shot 1 --stop_epoch 500 \
 --warmup baseline --train_aug --model ResNet10 --name ban_episodic_gen_3_run1 --gen 3 \
 --wandb --wandb_project_name ban_episodic_gen_3 --wandb_run_name run1 \
& \
CUDA_VISIBLE_DEVICES=2 python train_proto_gen_1.py --method protonet --dataset miniImagenet --testset cub --n_shot 1 --stop_epoch 500 \
 --warmup baseline --train_aug --model ResNet10 --name ban_episodic_gen_3_run2 --gen 3 \
 --wandb --wandb_project_name ban_episodic_gen_3 --wandb_run_name run2 \