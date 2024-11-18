set -x
port=$(shuf -i25000-30000 -n1)
TEST_JSON_DIR=data/zero-shot-test.jsonl
MODEL_NAME_OR_PATH=output/flan-t5-base-task-adaptation-1/checkpoint-4956
BEAM_SIZE=1
OUTPUT_DIR=output/flan-t5-xxl-task-adaptation-beam${BEAM_SIZE}
RUN_NAME=flan-t5-xxl-experiment
deepspeed --include="localhost:0,1,2,3,4,5,6,7" --master_port $port src/run.py \
    --do_predict --predict_with_generate \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --test_json_dir $TEST_JSON_DIR \
    --no_load_gner_customized_datasets \
    --output_dir $OUTPUT_DIR \
    --run_name $RUN_NAME \
    --generation_num_beams ${BEAM_SIZE} \
    --preprocessing_num_workers 4 \
    --per_device_eval_batch_size 64 \
    --bf16 True --tf32 True \
    --max_source_length 640 \
    --max_target_length 640 \
    --generation_max_length 640 \
    --overwrite_output_dir \
    --overwrite_cache \
    --seed 1234
