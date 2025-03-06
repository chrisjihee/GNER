#!/usr/bin/env bash
set -x

DEEPSPEED_CONFIG="configs/deepspeed/ds1_t5.json"
DEEPSPEED_PORT=$(shuf -i 25000-30000 -n 1)
CUDA_DEVICES=0
SOURCE_FILE="run_ner.py"
TRAIN_FILE="data/conll2003/train.json"
VALID_FILE="data/conll2003/validation.json"
OUTPUT_NAME="NER-CONLL"

MODEL_NAMES=(
  "google-bert/bert-base-cased"
#  "FacebookAI/roberta-base"
#  "FacebookAI/roberta-large"
#  "answerdotai/ModernBERT-base"
#  "answerdotai/ModernBERT-large"
#  "microsoft/deberta-v3-base"
#  "microsoft/deberta-v3-large"
#  "microsoft/deberta-v2-xlarge"
#  "microsoft/deberta-v2-xxlarge"
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
      --output_dir output/$OUTPUT_NAME/$MODEL_NAME \
      --cache_dir .cache \
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
      --overwrite_output_dir \
      --overwrite_cache \
      --eval_on_start
done
