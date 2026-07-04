#!/bin/bash
cd ~/TraceDet-


echo "Starting hallucination evaluation using Qwen8B..."

CUDA_VISIBLE_DEVICES=1 python ./DLM_generate/metric_qwen.py > logs/qwen_eval.log 2>&1


echo "All tasks completed."