#!/bin/bash
python3 recon_train.py \
        --shape_y 320 --shape_z 256 \
        --num_channels 8 --num_maps 1 \
        --batch_size 6 \
        --loss_l1 0.9 --loss_l2 0 --loss_adv 0.1 \
        --model_dir summary/knee_adv_ws \
        --warm_start_dir summary/knee_l1 \
        --shape_calib 10 \
        --max_steps 50000 \
        --learning_rate 1e-4 \
        --device 0
