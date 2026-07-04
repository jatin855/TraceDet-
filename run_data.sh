#!/bin/bash
cd ~/TraceDet-






CUDA_VISIBLE_DEVICES=0 python -m DLM_generate.generate_interoutput --dataset triviaqa --task train --generate_task entropy --gen_length 128
CUDA_VISIBLE_DEVICES=0 python -m DLM_generate.generate_interoutput --dataset triviaqa --task val --generate_task entropy --gen_length 128
CUDA_VISIBLE_DEVICES=0 python -m DLM_generate.generate_interoutput --dataset triviaqa --task test --generate_task entropy --gen_length 128



wait
