#!/usr/bin/env bash
set -x

CUDA_DEVICES=0
PROGRAM_SOURCE="run_glue.py"
DEEPSPEED_PORT=$(shuf -i 25000-30000 -n 1)
DEEPSPEED_CONFIG="configs/deepspeed/ds0_t5.json"

MODELS=(
  "google-bert/bert-base-cased"
  "google-bert/bert-large-cased"
  "FacebookAI/roberta-base"
  "FacebookAI/roberta-large"
  "FacebookAI/xlm-roberta-base"
  "FacebookAI/xlm-roberta-large"
  "microsoft/deberta-v3-base"
  "microsoft/deberta-v3-large"
  "answerdotai/ModernBERT-base"
  "answerdotai/ModernBERT-large"
)

for MODEL_NAME_OR_PATH in "${MODELS[@]}"; do
  MODEL_OUT_DIR=$(echo "$MODEL_NAME_OR_PATH" | tr '/' '-')

  python -m deepspeed.launcher.runner --include="localhost:$CUDA_DEVICES" --master_port "$DEEPSPEED_PORT" "$PROGRAM_SOURCE" \
    --model_name_or_path "$MODEL_NAME_OR_PATH" \
    --task_name stsb \
    --do_train \
    --do_eval \
    --bf16 True \
    --tf32 True \
    --max_seq_length 256 \
    --per_device_eval_batch_size 4 \
    --per_device_train_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 5 \
    --logging_strategy epoch \
    --eval_strategy epoch \
    --save_strategy no \
    --deepspeed "$DEEPSPEED_CONFIG" \
    --overwrite_cache \
    --overwrite_output_dir \
    --output_dir "output/stsb/$MODEL_OUT_DIR"
done
