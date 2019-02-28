#!/bin/bash
python3 recon_train.py \
        --shape_y 320 --shape_z 256 \
        --num_channels 8 --num_maps 1 \
        --batch_size 2 \
        --loss_l1 0 --loss_l2 1 \
        --model_dir summary/knee_l2 \
        --max_steps 10000 \
        --device 0
