set -x
port=$(shuf -i25000-30000 -n1)
DATA_DIR=data
TRAIN_JSON_DIR=data/pile-ner.json
DATA_CONFIG_DIR=configs/dataset_configs/task_adaptation_configs
INSTRUCTION_FILE=configs/instruction_configs/instruction.json
DEEPSPEED_CONFIG=configs/deepspeed_configs/deepspeed_zero0_t5.json
MODEL_NAME_OR_PATH=google/flan-t5-large
OUTPUT_DIR=output/flan-t5-large-task-adaptation
RUN_NAME=flan-t5-large-experiment
deepspeed --include="localhost:0,1,2,3,4,5,6,7" --master_port $port src/run.py \
    --bf16 True --tf32 True \
    --do_train \
    --do_predict \
    --predict_with_generate \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --data_dir $DATA_DIR \
    --preprocessing_num_workers 4 \
    --train_json_dir $TRAIN_JSON_DIR \
    --data_config_dir $DATA_CONFIG_DIR \
    --instruction_file $INSTRUCTION_FILE \
    --output_dir $OUTPUT_DIR \
    --per_device_eval_batch_size 32 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --gradient_checkpointing True \
    --lr_scheduler_type constant \
    --learning_rate 5e-05 \
    --warmup_steps 0 \
    --num_train_epochs 10 \
    --deepspeed $DEEPSPEED_CONFIG \
    --run_name $RUN_NAME \
    --max_source_length 640 \
    --max_target_length 640 \
    --generation_max_length 640 \
    --overwrite_output_dir \
    --logging_strategy steps \
    --logging_steps 10 \
    --eval_strategy no \
    --save_strategy epoch \
    --seed 1234
