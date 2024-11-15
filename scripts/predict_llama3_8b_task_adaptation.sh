set -x

port=$(shuf -i25000-30000 -n1)

MODEL_NAME_OR_PATH=meta-llama/Llama-3.1-8B
MODEL_CHECKPOINT_PATH=output/llama3-8b-task-adaptation/checkpoint-1236
DATA_DIR=data
DATA_CONFIG_DIR=configs/dataset_configs/task_adaptation_configs
INSTRUCTION_FILE=configs/instruction_configs/instruction.json
OUTPUT_DIR=output/llama3-8b-task-adaptation-predict
DEEPSPEED_CONFIG=configs/deepspeed_configs/deepspeed_zero3_llama.json
RUN_NAME=llama3-8B-experiment

deepspeed --include="localhost:0,1,2,3,4,5,6,7" --master_port $port src/run.py \
    --bf16 True --tf32 True \
    --do_predict \
    --predict_with_generate \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --resume_from_checkpoint $MODEL_CHECKPOINT_PATH \
    --data_dir $DATA_DIR \
    --preprocessing_num_workers 12 \
    --data_config_dir $DATA_CONFIG_DIR \
    --instruction_file $INSTRUCTION_FILE \
    --output_dir $OUTPUT_DIR \
    --per_device_eval_batch_size 4 \
    --deepspeed $DEEPSPEED_CONFIG \
    --run_name $RUN_NAME \
    --max_source_length 640 \
    --max_target_length 640 \
    --generation_max_length 1280 \
    --overwrite_output_dir \
    --overwrite_cache \
    --seed 1234
