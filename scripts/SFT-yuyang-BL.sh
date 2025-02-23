set -x
CUDA_VISIBLE_DEVICES=4,5,6,7
MASTER_PORT=$(shuf -i25000-30000 -n1)
RUN_NAME=SFT-yuyang-BL-$(hostname)
DATA_DIR=data
OUTPUT_DIR=output-lfs
DATA_CONFIG_DIR=configs/dataset/SFT
INSTRUCTION_FILE=configs/instruction/GNER-paper.json
DEEPSPEED_CONFIG=configs/deepspeed/ds2_t5.json
MODEL_NAME_OR_PATH=google/flan-t5-base


deepspeed --include="localhost:$CUDA_VISIBLE_DEVICES" --master_port $MASTER_PORT gner/run.py \
    --do_train \
    --do_predict \
    --predict_with_generate \
    --run_name $RUN_NAME \
    --data_dir $DATA_DIR \
    --output_dir $OUTPUT_DIR/$RUN_NAME \
    --data_config_dir $DATA_CONFIG_DIR \
    --instruction_file $INSTRUCTION_FILE \
    --deepspeed $DEEPSPEED_CONFIG \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --metric_for_best_model eval_average \
    --load_best_model_at_end True \
    --greater_is_better True \
    --gradient_accumulation_steps 8 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --preprocessing_num_workers 4 \
    --overwrite_output_dir \
    --overwrite_cache \
    --num_train_epochs 10 \
    --max_source_length 640 \
    --max_target_length 640 \
    --generation_max_length 640 \
    --lr_scheduler_type constant \
    --learning_rate 5e-05 \
    --warmup_steps 0 \
    --logging_steps 10 \
    --logging_strategy steps \
    --eval_strategy epoch \
    --save_strategy epoch \
    --seed 1234
