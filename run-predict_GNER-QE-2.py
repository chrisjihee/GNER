import os
import random
import subprocess

from chrisbase.io import dirs, files

DEEPSPEED_CONFIG = "configs/deepspeed/ds3_t5.json"
DEEPSPEED_PORT = random.randint(25000, 30000)
CUDA_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES", "2")
SOURCE_FILE = "run_glue.py"

# trained_model_paths = dirs("output-lfs/GNER-QE-HR-*/**/roberta-*/checkpoint-*")
trained_model_paths = dirs("output-lfs/GNER-QE-HR-*/**/deberta-*/checkpoint-*")
TRAIN_DATA_POST = "max_sampled=3"
TRAIN_FILE = f"data/GNER-QE/pile-ner-sampled-N19988-quality_est-${TRAIN_DATA_POST}.json"
VALID_FILE = "data/GNER-QE/ZSE-validation-sampled-N210-quality_est-max_sampled=0.json"
TEST_FILE = "data/GNER-QE/ZSE-test-sampled-N700-quality_est-max_sampled=0.json"

for model_path in trained_model_paths:
    command = f"""
        python -m
            deepspeed.launcher.runner
                --include=localhost:{CUDA_DEVICES}
                --master_port={DEEPSPEED_PORT}
            {SOURCE_FILE}
                --train_file {TRAIN_FILE}
                --validation_file {VALID_FILE}
                --test_file {TEST_FILE}
                --model_name_or_path {model_path}
                --output_dir {model_path}/pred
                --cache_dir .cache
                --do_predict
                --bf16 True --tf32 True
                --max_seq_length 640
                --per_device_eval_batch_size 30
                --per_device_train_batch_size 16
                --gradient_accumulation_steps 1
                --deepspeed {DEEPSPEED_CONFIG}
                --logging_strategy epoch
                --eval_strategy epoch
                --save_strategy epoch
                --overwrite_output_dir
    """
    command = command.strip().split()
    print("*" * 120)
    print("[COMMAND]", " ".join(command))
    print("*" * 120)

    subprocess.run(command)
    print("\n" * 3)
