#!/bin/bash

# 设置环境变量
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_OFFLINE="1"

# 运行脚本
python -u generate.py \
  --model_name Qwen/Qwen3-4B \
  --decoding_strategy sample \
  --output_dir outputs \
  --dataset_name gsm8k\
  --data_path gsm8k/grade_school_math/data/train.jsonl
  --max_samples 0\
  --question_key question