#!/bin/bash
device=0

# L1-Loss with 10,000 steps
echo ">>> Training L1-Loss..."
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
python3 recon_train.py \
        --shape_y 320 --shape_z 256 \
        --num_channels 8 --num_maps 1 \
        --batch_size 2 \
        --loss_l1 0.9 --loss_l2 0 --loss_adv 0.1 \
        --model_dir summary/knee_adv_wsl1 \
        --max_steps 60000 \
        --device $device

# Finish up L1-Loss training
python3 recon_train.py \
        --shape_y 320 --shape_z 256 \
        --num_channels 8 --num_maps 1 \
        --batch_size 2 \
        --model_dir summary/knee_l1 \
        --loss_l1 1 \
        --max_steps 20000 \
        --device $device

# Train with L2-Loss
python3 recon_train.py \
        --shape_y 320 --shape_z 256 \
        --num_channels 8 --num_maps 1 \
        --batch_size 2 \
        --model_dir summary/knee_l2 \
        --loss_l1 0 --loss_l2 1 \
        --max_steps 20000 \
        --device $device
