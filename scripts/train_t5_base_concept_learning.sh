set -x
port=$(shuf -i25000-30000 -n1)
TRAIN_JSON_DIR=data/pile-ner-with-wiki.jsonl
VALID_JSON_DIR=data/zero-shot-with-wiki.jsonl
TEST_JSON_DIR=data/zero-shot-with-wiki.jsonl
DEEPSPEED_CONFIG=configs/deepspeed_configs/deepspeed_zero0_t5.json
MODEL_NAME_OR_PATH=google/flan-t5-base
OUTPUT_DIR=output/flan-t5-base-concept-learning
RUN_NAME=flan-t5-base-experiment
deepspeed --include="localhost:0,1,2,3,4,5,6,7" --master_port $port src/run.py \
    --do_train --do_predict --predict_with_generate \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --train_json_dir $TRAIN_JSON_DIR \
    --valid_json_dir $VALID_JSON_DIR \
    --test_json_dir $TEST_JSON_DIR \
    --no_load_gner_customized_datasets \
    --output_dir $OUTPUT_DIR \
    --run_name $RUN_NAME \
    --preprocessing_num_workers 4 \
    --per_device_eval_batch_size 32 \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --gradient_checkpointing True \
    --bf16 True --tf32 True \
    --lr_scheduler_type constant \
    --learning_rate 5e-05 \
    --warmup_steps 0 \
    --num_train_epochs 24 \
    --max_source_length 640 \
    --max_target_length 640 \
    --generation_max_length 1280 \
    --logging_strategy steps --logging_steps 10 \
    --save_strategy epoch \
    --eval_strategy epoch \
    --overwrite_output_dir \
    --overwrite_cache \
    --load_best_model_at_end True \
    --metric_for_best_model eval_average_f1 --greater_is_better True \
    --seed 1234 \
    --deepspeed $DEEPSPEED_CONFIG  # --max_train_samples 10240 --max_eval_samples 1024 --max_predict_samples 1024

# + deepspeed --include=localhost:0,1,2,3,4,5,6,7 --master_port 29048 src/run.py --do_train --do_predict --predict_with_generate --model_name_or_path google/flan-t5-base --data_dir data --preprocessing_num_workers 4 --train_json_dir data/pile-ner.json --data_config_dir configs/dataset_configs/task_adaptation_configs --instruction_file configs/instruction_configs/instruction.json --output_dir output/flan-t5-base-task-adaptation --per_device_eval_batch_size 32 --per_device_train_batch_size 32 --gradient_accumulation_steps 1 --gradient_checkpointing True --bf16 True --tf32 True --lr_scheduler_type constant --learning_rate 5e-05 --warmup_steps 0 --num_train_epochs 12 --run_name flan-t5-base-experiment --max_source_length 640 --max_target_length 640 --generation_max_length 640 --logging_strategy steps --logging_steps 10 --save_strategy epoch --eval_strategy epoch --overwrite_output_dir --overwrite_cache --load_best_model_at_end True --metric_for_best_model eval_average_f1 --greater_is_better True --seed 1234 --deepspeed configs/deepspeed_configs/deepspeed_zero0_t5.json