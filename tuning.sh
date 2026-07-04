#!/bin/bash
cd ~/TraceDet-

#CUDA_VISIBLE_DEVICES=1  python train_tuning.py --num_repeats 3 --device cuda --dataset commonsenseqa   > logs/commonsenseqa_l2_tuning.log 2>&1 &
CUDA_VISIBLE_DEVICES=1  python train_tuning.py --num_repeats 3 --device cuda --dataset triviaqa  
#CUDA_VISIBLE_DEVICES=1  python train_tuning.py --num_repeats 3 --device cuda --dataset hotpotqa   > logs/hotpotqa_l2_tuning.log 2>&1 &


wait