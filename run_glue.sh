#!/usr/bin/env bash
set -x

DEEPSPEED_CONFIG="configs/deepspeed/ds1_t5.json"
DEEPSPEED_PORT=$(shuf -i 25000-30000 -n 1)
CUDA_DEVICES=0,1
PROGRAM_SOURCE="run_glue.py"
OUTPUT_NAME="GLUE-STSb"
TASK_NAME="stsb"

MODEL_NAMES=(
  "google-bert/bert-base-cased"
  "FacebookAI/roberta-base"
  "FacebookAI/roberta-large"
  "answerdotai/ModernBERT-base"
  "answerdotai/ModernBERT-large"
  "microsoft/deberta-v3-base"
  "microsoft/deberta-v3-large"
  "microsoft/deberta-v2-xlarge"
  "microsoft/deberta-v2-xxlarge"
)

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
  python -m deepspeed.launcher.runner --include=localhost:$CUDA_DEVICES --master_port $DEEPSPEED_PORT $PROGRAM_SOURCE \
    --model_name_or_path $MODEL_NAME \
    --task_name stsb \
    --do_train \
    --do_eval \
    --bf16 True \
    --tf32 True \
    --max_seq_length 256 \
    --per_device_eval_batch_size 4 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 4 \
    --learning_rate 2e-5 \
    --num_train_epochs 10 \
    --logging_strategy epoch \
    --eval_strategy epoch \
    --save_strategy no \
    --deepspeed $DEEPSPEED_CONFIG \
    --output_dir output/$OUTPUT_NAME/$MODEL_NAME_OR_PATH \
    --overwrite_output_dir \
    --overwrite_cache
done
