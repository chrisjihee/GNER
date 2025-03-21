#!/usr/bin/env bash
set -x

DEEPSPEED_PORT=$(shuf -i 25000-30000 -n 1)
CUDA_DEVICES=0,1,2,3
SOURCE_FILE="run_glue.py"
TRAIN_EPOCHS=1
TRAIN_DATA_POST="max_sampled=3"
TRAIN_FILE="data/GNER-QE/pile-ner-sampled-N19988-quality_est-${TRAIN_DATA_POST}.json"
VALID_FILE="data/GNER-QE/ZSE-validation-sampled-N210-quality_est-max_sampled=0.json"
TEST_FILE="data/GNER-QE/ZSE-test-sampled-N700-quality_est-max_sampled=0.json"
OUTPUT_NAME="GNER-QE-HR-${TRAIN_DATA_POST}"
OUTPUT_HOME="output-lfs"

MODEL_NAMES=(
#  "FacebookAI/roberta-base"
  "FacebookAI/roberta-large"
)

LEARNING_RATES=(
  1e-5
  2e-5
  3e-5
  4e-5
  5e-5
)

for LR in "${LEARNING_RATES[@]}"; do
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
        --output_dir ${OUTPUT_HOME}/${OUTPUT_NAME}/${MODEL_NAME}-ep${TRAIN_EPOCHS}-lr${LR} \
        --cache_dir .cache \
        --do_train --do_eval --do_predict \
        --bf16 True --tf32 True \
        --max_seq_length 512 \
        --per_device_eval_batch_size 32 \
        --per_device_train_batch_size 16 \
        --gradient_accumulation_steps 1 \
        --deepspeed "configs/deepspeed/ds0_t5.json" \
        --learning_rate $LR \
        --logging_steps 50 \
        --eval_steps 500 \
        --save_steps 500 \
        --max_steps 7000 \
        --num_train_epochs $TRAIN_EPOCHS \
        --logging_strategy steps \
        --eval_strategy steps \
        --save_strategy steps \
        --save_total_limit 2 \
        --load_best_model_at_end True \
        --metric_for_best_model pearson \
        --overwrite_output_dir
  done
done
