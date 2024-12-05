set -x
port=$(shuf -i25000-30000 -n1)
TRAIN_JSON_DIR=data/zero-shot-train-plus.jsonl
VALID_JSON_DIR=data/zero-shot-test.jsonl
MODEL_NAME_OR_PATH=meta-llama/Llama-3.2-3B
OUTPUT_DIR=output/train_llama3_3b_supervised-plus
RUN_NAME=train_llama3_3b_supervised-plus
DEEPSPEED_CONFIG=configs/deepspeed_configs/deepspeed_zero2_llama.json
deepspeed --include="localhost:0,1,2,3,4,5,6,7" --master_port $port src/run.py \
    --do_train --do_eval --predict_with_generate \
    --train_json_dir $TRAIN_JSON_DIR \
    --valid_json_dir $VALID_JSON_DIR \
    --no_load_gner_customized_datasets \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --output_dir $OUTPUT_DIR \
    --run_name $RUN_NAME \
    --preprocessing_num_workers 4 \
    --per_device_eval_batch_size 32 \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing True \
    --bf16 True --tf32 True \
    --lr_scheduler_type cosine \
    --learning_rate 2e-05 \
    --warmup_ratio 0.04 --weight_decay 0. \
    --num_train_epochs 6 \
    --max_source_length 640 \
    --max_target_length 640 \
    --generation_max_length 1280 \
    --logging_strategy steps --logging_steps 10 \
    --eval_strategy epoch \
    --save_strategy no \
    --overwrite_output_dir --overwrite_cache \
    --seed 1234 --deepspeed $DEEPSPEED_CONFIG
