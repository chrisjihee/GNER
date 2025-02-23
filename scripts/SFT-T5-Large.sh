set -x
CUDA_VISIBLE_DEVICES=4,5,6,7
MASTER_PORT=$(shuf -i25000-30000 -n1)
DATA_DIR=data
INSTRUCTION_FILE=configs/instruction/GNER-paper.json
DEEPSPEED_CONFIG=configs/deepspeed/ds2_t5.json
MODEL_NAME_OR_PATH=google/flan-t5-large
OUTPUT_DIR=output/SFT-T5-Large
RUN_NAME=SFT-T5-Large
DATA_CONFIG_DIR=configs/dataset/SFT


deepspeed --include="localhost:$CUDA_VISIBLE_DEVICES" --master_port $MASTER_PORT gner/run.py \
    --do_train \
    --do_predict \
    --predict_with_generate \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --data_dir $DATA_DIR \
    --preprocessing_num_workers 4 \
    --load_best_model_at_end True \
    --metric_for_best_model "eval_average_f1" \
    --greater_is_better True \
    --instruction_file $INSTRUCTION_FILE \
    --data_config_dir $DATA_CONFIG_DIR \
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
