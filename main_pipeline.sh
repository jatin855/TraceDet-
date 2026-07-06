#!/bin/bash
set -e
cd ~/TraceDet-

mkdir -p logs

echo "=========================================="
echo "STEP 1: Generating LLM outputs + entropy traces"
echo "=========================================="
bash run_data.sh
echo "Step 1 complete: $(date)"

echo "=========================================="
echo "STEP 2: Running Qwen hallucination judging"
echo "=========================================="
bash qwen_eval.sh
echo "Step 2 complete: $(date)"

echo "=========================================="
echo "STEP 3: Building train/val/test split file"
echo "=========================================="
python make_split.py
echo "Step 3 complete: $(date)"

echo "=========================================="
echo "STEP 4: Training TimeHalu detector"
echo "=========================================="
python train_main.py --dataset triviaqa --task entropy --device cuda
echo "Step 4 complete: $(date)"

echo "=========================================="
echo "PIPELINE FINISHED SUCCESSFULLY: $(date)"
