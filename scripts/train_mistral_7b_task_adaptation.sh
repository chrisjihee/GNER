set -x
port=$(shuf -i25000-30000 -n1)
DATA_DIR=data
TRAIN_JSON_DIR=data/pile-ner.json
INSTRUCTION_FILE=configs/instruction_configs/instruction.json
DATA_CONFIG_DIR=configs/dataset_configs/task_adaptation_configs
DEEPSPEED_CONFIG=configs/deepspeed_configs/deepspeed_zero2_llama.json
MODEL_NAME_OR_PATH=mistralai/Mistral-7B-Instruct-v0.2
OUTPUT_DIR=output/mistral-7b-task-adaptation
RUN_NAME=mistral-7B-experiment
deepspeed --include="localhost:0,1,2,3,4,5,6,7" --master_port $port src/run.py \
    --bf16 True --tf32 True \
    --do_train \
    --do_predict \
    --predict_with_generate \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --data_dir $DATA_DIR \
    --preprocessing_num_workers 12 \
    --train_json_dir $TRAIN_JSON_DIR \
    --data_config_dir $DATA_CONFIG_DIR \
    --instruction_file $INSTRUCTION_FILE \
    --output_dir $OUTPUT_DIR \
    --per_device_eval_batch_size 40 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 32 \
    --gradient_checkpointing True \
    --lr_scheduler_type cosine \
    --learning_rate 2e-05 \
    --warmup_ratio 0.04 \
    --weight_decay 0. \
    --num_train_epochs 3 \
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