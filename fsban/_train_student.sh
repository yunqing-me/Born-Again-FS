#!/usr/bin/env bash

export PYTHONPATH=.






CUDA_VISIBLE_DEVICES=0 python train.py --method matchingnet --dataset multi --testset cub --warmup baseline++_conv4 --train_aug --model Conv4 --name fsban_matchingnet_cub_shot_5_conv4_teacher_resnet10 --teacher ResNet10 --n_shot 5 --stop_epoch 600 \
& \
CUDA_VISIBLE_DEVICES=1 python train.py --method matchingnet --dataset multi --testset cars --warmup baseline++_conv4 --train_aug --model Conv4 --name fsban_matchingnet_cars_shot_5_conv4_teacher_resnet10 --teacher ResNet10 --n_shot 5 --stop_epoch 600 \
& \
CUDA_VISIBLE_DEVICES=2 python train.py --method matchingnet --dataset multi --testset places --warmup baseline++_conv4 --train_aug --model Conv4 --name fsban_matchingnet_places_shot_5_conv4_teacher_resnet10 --teacher ResNet10 --n_shot 5 --stop_epoch 600 \
& \
CUDA_VISIBLE_DEVICES=3 python train.py --method matchingnet --dataset multi --testset plantae --warmup baseline++_conv4 --train_aug --model Conv4 --name fsban_matchingnet_plantae_shot_5_conv4_teacher_resnet10 --teacher ResNet10 --n_shot 5 --stop_epoch 600 \
& \
# CUDA_VISIBLE_DEVICES=4 python train.py --method matchingnet --dataset multi --testset cub --warmup baseline++_conv4 --train_aug --model Conv4 --name fsban_matchingnet_cub_shot_5_conv4_teacher_conv4 --teacher Conv4 --n_shot 5 --stop_epoch 400 \
# & \
# CUDA_VISIBLE_DEVICES=5 python train.py --method matchingnet --dataset multi --testset cars --warmup baseline++_conv4 --train_aug --model Conv4 --name fsban_matchingnet_cars_shot_5_conv4_teacher_conv4 --teacher Conv4 --n_shot 5 --stop_epoch 400 \
# & \
# CUDA_VISIBLE_DEVICES=6 python train.py --method matchingnet --dataset multi --testset places --warmup baseline++_conv4 --train_aug --model Conv4 --name fsban_matchingnet_places_shot_5_conv4_teacher_conv4 --teacher Conv4 --n_shot 5 --stop_epoch 400 \
# & \
# CUDA_VISIBLE_DEVICES=7 python train.py --method matchingnet --dataset multi --testset plantae --warmup baseline++_conv4 --train_aug --model Conv4 --name fsban_matchingnet_plantae_shot_5_conv4_teacher_conv4 --teacher Conv4 --n_shot 5 --stop_epoch 400 \
