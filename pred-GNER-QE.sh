#!/usr/bin/env bash
set -x

DEEPSPEED_CONFIG="configs/deepspeed/ds3_t5.json"
DEEPSPEED_PORT=$(shuf -i 25000-30000 -n 1)
CUDA_DEVICES=0,1
SOURCE_FILE="run_glue.py"
TRAIN_FILE="data/GNER-QE/ZSE-validation-pred-by_beam-num=30-train.json"
VALID_FILE="data/GNER-QE/ZSE-validation-pred-by_beam-num=30-val.json"
TEST_FILES=(
  "data/GNER-QE/ZSE-validation-pred-by_beam-num=10-val.json"
  "data/GNER-QE/ZSE-validation-pred-by_beam-num=20-val.json"
  "data/GNER-QE/ZSE-validation-pred-by_beam-num=30-val.json"
  "data/GNER-QE/ZSE-validation-pred-by_beam-num=40-val.json"
  "data/GNER-QE/ZSE-validation-pred-by_beam-num=50-val.json"
)

MODEL_NAMES=(
  "output/GNER-QE/google-bert/bert-base-cased-num=30/checkpoint-29448"
  "output/GNER-QE/google-bert/bert-base-cased-num=30/checkpoint-26994"
  "output/GNER-QE/google-bert/bert-base-cased-num=40/checkpoint-62092"
  "output/GNER-QE/google-bert/bert-base-cased-num=20/checkpoint-27880"
  "output/GNER-QE/FacebookAI/roberta-large-num=10/checkpoint-15244"
)

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
  for TEST_FILE in "${TEST_FILES[@]}"; do
    python -m \
      deepspeed.launcher.runner \
        --include=localhost:$CUDA_DEVICES \
        --master_port $DEEPSPEED_PORT \
      $SOURCE_FILE \
        --train_file $TRAIN_FILE \
        --validation_file $VALID_FILE \
        --test_file $TEST_FILE \
        --model_name_or_path $MODEL_NAME \
        --output_dir $MODEL_NAME-pred \
        --cache_dir .cache \
        --do_predict \
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
done
