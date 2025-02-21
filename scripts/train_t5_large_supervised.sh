set -x

port=$(shuf -i25000-30000 -n1)

MODEL_NAME_OR_PATH=google/flan-t5-large
DATA_DIR=data
DATA_CONFIG_DIR=configs/dataset/supervised
INSTRUCTION_FILE=configs/instruction/GNER-paper.json
OUTPUT_DIR=output/flan-t5-large-supervised
DEEPSPEED_CONFIG=configs/deepspeed/ds0_t5.json

RUN_NAME=flan-t5-large-experiment

deepspeed --include="localhost:4,5,6,7" --master_port $port gner/run.py \
    --do_train \
    --do_predict \
    --predict_with_generate \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --data_dir $DATA_DIR \
    --preprocessing_num_workers 4 \
    --load_best_model_at_end True \
    --metric_for_best_model "eval_average_f1" \
    --greater_is_better True \
    --data_config_dir $DATA_CONFIG_DIR \
    --instruction_file $INSTRUCTION_FILE \
    --output_dir $OUTPUT_DIR \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-05 \
    --num_train_epochs 10 \
    --deepspeed $DEEPSPEED_CONFIG \
    --run_name $RUN_NAME \
    --max_source_length 640 \
    --max_target_length 640 \
    --generation_max_length 640 \
    --overwrite_output_dir \
    --overwrite_cache \
    --lr_scheduler_type constant \
    --warmup_steps 0 \
    --logging_strategy steps \
    --logging_steps 10 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --seed 1234
