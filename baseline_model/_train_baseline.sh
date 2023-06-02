#!/usr/bin/env bash

export PYTHONPATH=.



# for baseline and baseline++
# train rfs feature encoder - on multiple source domains, with multi-head classifier
# mini-ImageNet is always seen domain
# CUDA_VISIBLE_DEVICES=0 python train_distill_rfs.py --method baseline++ --dataset multi --testset cub --name baseline++_seen_multi_unseen_cub_gen_1 --train_aug --model ResNet10 \
# & \
# CUDA_VISIBLE_DEVICES=1 python train_distill_rfs.py --method baseline++ --dataset multi --testset cars --name baseline++_seen_multi_unseen_cars_gen_1 --train_aug --model ResNet10 \
# & \
# CUDA_VISIBLE_DEVICES=2 python train_distill_rfs.py --method baseline++ --dataset multi --testset places --name baseline++_seen_multi_unseen_places_gen_1 --train_aug --model ResNet10 \
# & \
# CUDA_VISIBLE_DEVICES=3 python train_distill_rfs.py --method baseline++ --dataset multi --testset plantae --name baseline++_seen_multi_unseen_plantae_gen_1 --train_aug --model ResNet10 \


CUDA_VISIBLE_DEVICES=1 python train_baseline.py --method baseline --name baseline_conv4 --model Conv4 --dataset miniImagenet --stop_epoch 400 --train_aug