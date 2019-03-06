#!/bin/bash
device=0

echo ">>> Training L1 loss with 10k steps..."
python3 recon_train.py \
        --shape_y 320 --shape_z 256 \
        --num_channels 8 --num_maps 1 \
        --batch_size 2 \
        --model_dir summary/knee_l1 \
        --loss_l1 1 \
        --max_steps 10000 \
        --device $device

# Warm-start adversarial loss with L1-Loss training
# 60,000 steps: train recon network 1 time for every 5 steps of adversarial network
echo ">>> Training with adversarial + L1 loss with 60k steps..."
python3 recon_train.py \
        --shape_y 320 --shape_z 256 \
        --num_channels 8 --num_maps 1 \
        --batch_size 2 \
        --loss_l1 0.9 --loss_l2 0 --loss_adv 0.1 \
        --model_dir summary/knee_adv_wsl1 \
        --warm_start_dir summary/knee_l1 \
        --max_steps 60000 \
        --learning_rate 1e-4 \
        --device $device

echo ">>> Finishing training with L1 loss with +10k steps..."
python3 recon_train.py \
        --shape_y 320 --shape_z 256 \
        --num_channels 8 --num_maps 1 \
        --batch_size 2 \
        --model_dir summary/knee_l1 \
        --loss_l1 1 \
        --max_steps 20000 \
        --device $device

echo ">>> Training L2 loss with 20k steps..."
python3 recon_train.py \
        --shape_y 320 --shape_z 256 \
        --num_channels 8 --num_maps 1 \
        --batch_size 2 \
        --model_dir summary/knee_l2 \
        --loss_l1 0 --loss_l2 1 \
        --max_steps 20000 \
        --device $device
