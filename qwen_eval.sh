#!/bin/bash
cd ~/TraceDet-


echo "Starting hallucination evaluation using Qwen8B..."

CUDA_VISIBLE_DEVICES=1 python ./DLM_generate/metric_qwen.py 

echo "All tasks completed."