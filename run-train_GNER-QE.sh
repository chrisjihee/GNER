#!/usr/bin/env bash
set -x

DEEPSPEED_CONFIG="configs/deepspeed/ds1_t5.json"
DEEPSPEED_PORT=$(shuf -i 25000-30000 -n 1)
CUDA_DEVICES=0,1,2,3
SOURCE_FILE="run_glue.py"
TRAIN_POSTFIX="max_sampled=3"
TRAIN_FILE="data/GNER-QE/pile-ner-sampled-N19988-quality_est-${TRAIN_POSTFIX}.json"
VALID_FILE="data/GNER-QE/ZSE-validation-sampled-N210-quality_est-max_sampled=10.json"
TEST_FILE="data/GNER-QE/ZSE-test-sampled-N700-quality_est-max_sampled=0.json"
OUTPUT_NAME="GNER-QE-HR-ep1"
OUTPUT_HOME="output-lfs"

MODEL_NAMES=(
#  "FacebookAI/roberta-base"
#  "microsoft/deberta-v3-base"
  "FacebookAI/roberta-large"
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
      --per_device_eval_batch_size 64 \
      --per_device_train_batch_size 64 \
      --gradient_accumulation_steps 1 \
      --learning_rate 2e-5 \
      --num_train_epochs 1 \
      --logging_strategy steps \
      --eval_strategy steps \
      --save_strategy steps \
      --logging_steps 10 \
      --eval_steps 100 \
      --save_steps 100 \
      --save_total_limit 3 \
      --load_best_model_at_end True \
      --metric_for_best_model pearson \
      --deepspeed $DEEPSPEED_CONFIG \
      --overwrite_output_dir \
      --overwrite_cache
done
