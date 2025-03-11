#!/usr/bin/env bash
set -x

DEEPSPEED_CONFIG="configs/deepspeed/ds1_t5.json"
DEEPSPEED_PORT=$(shuf -i 25000-30000 -n 1)
CUDA_DEVICES=4,5
SOURCE_FILE="run_glue.py"
TRAIN_FILE="data/GNER-QE/ZSE-validation-pred-by_beam-num=30-train.json"
VALID_FILE="data/GNER-QE/ZSE-validation-pred-by_beam-num=30-val.json"
OUTPUT_NAME="GNER-QE"

MODEL_NAMES=(
#  "google-bert/bert-base-cased"
#  "FacebookAI/roberta-base"
#  "FacebookAI/roberta-large"
#  "answerdotai/ModernBERT-base"
#  "answerdotai/ModernBERT-large"
  "microsoft/deberta-v3-base"
  "microsoft/deberta-v3-large"
)

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
  python -m \
    deepspeed.launcher.runner \
      --include=localhost:$CUDA_DEVICES \
      --master_port $DEEPSPEED_PORT \
    $SOURCE_FILE \
      --train_file $TRAIN_FILE \
      --validation_file $VALID_FILE \
      --model_name_or_path $MODEL_NAME \
      --output_dir output/$OUTPUT_NAME/$MODEL_NAME-num=30 \
      --cache_dir .cache \
      --do_train \
      --do_eval \
      --bf16 True \
      --tf32 True \
      --max_seq_length 512 \
      --per_device_eval_batch_size 4 \
      --per_device_train_batch_size 4 \
      --gradient_accumulation_steps 4 \
      --learning_rate 2e-5 \
      --num_train_epochs 40 \
      --logging_strategy epoch \
      --eval_strategy epoch \
      --save_strategy epoch \
      --deepspeed $DEEPSPEED_CONFIG \
      --overwrite_output_dir \
      --overwrite_cache
done
