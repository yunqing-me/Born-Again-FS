#!/usr/bin/env bash

export PYTHONPATH=.


# CUDA_VISIBLE_DEVICES=2 python train_baseline_distill_gen_1_single.py --method baseline --dataset miniImagenet --gen 1 --stop_epoch 400 \
#  --warmup baseline --train_aug --model ResNet10 --name baseline_gen_1 \

# CUDA_VISIBLE_DEVICES=4 python train_baseline_distill_gen_1_single.py --method baseline --dataset miniImagenet --gen 2 --stop_epoch 400 \
#  --warmup baseline --train_aug --model ResNet10 --name baseline_gen_2 \
