#!/bin/bash
cd ~/TraceDet-



CUDA_VISIBLE_DEVICES=1 python -m DLM_generate.generate_interoutput --dataset triviaqa --task train --generate_task entropy --gen_length 128 > logs/triviaqa_train_ent.log 2>&1
CUDA_VISIBLE_DEVICES=0 python -m DLM_generate.generate_interoutput --dataset triviaqa --task val --generate_task entropy --gen_length 128 > logs/triviaqa_val_ent.log 2>&1
CUDA_VISIBLE_DEVICES=0 python -m DLM_generate.generate_interoutput --dataset triviaqa --task test --generate_task entropy --gen_length 128 > logs/triviaqa_test_ent.log 2>&1




wait
