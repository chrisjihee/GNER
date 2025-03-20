#!/usr/bin/env bash
set -x

DEEPSPEED_CONFIG="configs/deepspeed/ds1_t5.json"
DEEPSPEED_PORT=$(shuf -i 25000-30000 -n 1)
CUDA_DEVICES=6,7
SOURCE_FILE="run_glue.py"
TRAIN_POSTFIX="max_sampled=3"
TRAIN_FILE="data/GNER-QE/pile-ner-sampled-N19988-quality_est-max_sampled=3.jsonl"
VALID_FILE="data/GNER-QE/ZSE-validation-sampled-N210-quality_est-max_sampled=10.jsonl"
TEST_FILE="data/GNER-QE/ZSE-test-sampled-N700-quality_est-max_sampled=0.jsonl"
OUTPUT_NAME="GNER-QE-HR"
OUTPUT_HOME="output-lfs"

MODEL_NAMES=(
#  "FacebookAI/roberta-base"
#  "FacebookAI/roberta-large"
  "microsoft/deberta-v3-base"
#  "microsoft/deberta-v3-large"
)

for MODEL_NAME in "${MODEL_NAMES[@]}"; do
  python -m \
    deepspeed.launcher.runner \
      --include=localhost:$CUDA_DEVICES \
      --master_port $DEEPSPEED_PORT \
    $SOURCE_FILE \
      --train_file $TRAIN_FILE \
      --validation_file $VALID_FILE \
      --test_file $TEST_FILE \
      --model_name_or_path $MODEL_NAME \
      --output_dir $OUTPUT_HOME/$OUTPUT_NAME/$MODEL_NAME-$TRAIN_POSTFIX \
      --cache_dir .cache \
      --do_train \
      --do_eval \
      --do_predict \
      --bf16 True \
      --tf32 True \
      --max_seq_length 512 \
      --per_device_eval_batch_size 16 \
      --per_device_train_batch_size 4 \
      --gradient_accumulation_steps 4 \
      --learning_rate 2e-5 \
      --num_train_epochs 40 \
      --logging_strategy epoch \
      --eval_strategy epoch \
      --save_strategy epoch \
      --save_total_limit 3 \
      --load_best_model_at_end True \
      --metric_for_best_model pearson \
      --deepspeed $DEEPSPEED_CONFIG \
      --overwrite_output_dir \
      --overwrite_cache
done
