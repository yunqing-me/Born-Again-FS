#!/usr/bin/env bash

export PYTHONPATH=.





# CUDA_VISIBLE_DEVICES=0 python train_baseline.py --method baseline++ --dataset miniImagenet --name baseline++_conv6 --train_aug --model Conv6 \

CUDA_VISIBLE_DEVICES=0 python train_baseline.py --method matchingnet --dataset miniImagenet --testset miniImagenet --warmup baseline++_conv6 --train_aug --model Conv6 --name single_matchingnet_miniImagenet_shot_5_conv6 --n_shot 5 --stop_epoch 600 \
& \
CUDA_VISIBLE_DEVICES=1 python train_baseline.py --method matchingnet --dataset cub --testset cub --warmup baseline++_conv6 --train_aug --model Conv6 --name single_matchingnet_cub_shot_5_conv6 --n_shot 5 --stop_epoch 600 \
& \
CUDA_VISIBLE_DEVICES=2 python train_baseline.py --method matchingnet --dataset cars --testset cars --warmup baseline++_conv6 --train_aug --model Conv6 --name single_matchingnet_cars_shot_5_conv6 --n_shot 5 --stop_epoch 600 \
& \
CUDA_VISIBLE_DEVICES=3 python train_baseline.py --method matchingnet --dataset places --testset places --warmup baseline++_conv6 --train_aug --model Conv6 --name single_matchingnet_places_shot_5_conv6 --n_shot 5 --stop_epoch 600 \
& \
CUDA_VISIBLE_DEVICES=4 python train_baseline.py --method matchingnet --dataset plantae --testset plantae --warmup baseline++_conv6 --train_aug --model Conv6 --name single_matchingnet_plantae_shot_5_conv6 --n_shot 5 --stop_epoch 600 \
