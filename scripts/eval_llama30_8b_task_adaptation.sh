set -x
port=$(shuf -i25000-30000 -n1)
DATA_DIR=data
DATA_CONFIG_DIR=configs/dataset_configs/task_adaptation_configs
INSTRUCTION_FILE=configs/instruction_configs/instruction.json
BEAM_SIZE=1
MODEL_NAME_OR_PATH=output/llama30-8b-sta-task-adaptation/checkpoint-1236
OUTPUT_DIR=output/llama30-8b-task-adaptation-beam${BEAM_SIZE}
RUN_NAME=llama30-8b-experiment
deepspeed --include="localhost:0,1,2,3,4,5,6,7" --master_port $port gner/run.py \
    --do_predict --predict_with_generate \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --data_dir $DATA_DIR \
    --data_config_dir $DATA_CONFIG_DIR \
    --instruction_file $INSTRUCTION_FILE \
    --output_dir $OUTPUT_DIR \
    --run_name $RUN_NAME \
    --generation_num_beams ${BEAM_SIZE} \
    --preprocessing_num_workers 4 \
    --per_device_eval_batch_size 64 \
    --bf16 True --tf32 True \
    --max_source_length 640 \
    --max_target_length 640 \
    --generation_max_length 1280 \
    --overwrite_output_dir \
    --overwrite_cache \
    --seed 1234
