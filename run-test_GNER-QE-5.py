import os
import random
import subprocess

from chrisbase.io import dirs, files

DEEPSPEED_CONFIG = "configs/deepspeed/ds3_t5.json"
DEEPSPEED_PORT = random.randint(25000, 30000)
CUDA_DEVICES = os.getenv("CUDA_VISIBLE_DEVICES", "5")
SOURCE_FILE = "run_glue.py"

trained_model_paths = dirs("output/GNER-QE/**/deberta-v3-base-*/checkpoint-*")
train_files = files("data/GNER-QE/ZSE-validation-pred-by_beam-num=*-train.json")
valid_files = files("data/GNER-QE/ZSE-validation-pred-by_beam-num=*-val.json")
test_files = files("data/GNER-QE/ZSE-test-pred-by_beam-num=*-test.json")
assert len(train_files) == len(valid_files) == len(test_files), f"train_files: {len(train_files)}, valid_files: {len(valid_files)}, test_files: {len(test_files)}"

for model_path in trained_model_paths:
    for train_file, valid_file, test_file in zip(train_files, valid_files, test_files):
        command = f"""
            python -m
                deepspeed.launcher.runner
                    --include=localhost:{CUDA_DEVICES}
                    --master_port={DEEPSPEED_PORT}
                {SOURCE_FILE}
                    --train_file {train_file}
                    --validation_file {valid_file}
                    --test_file {test_file}
                    --model_name_or_path {model_path}
                    --output_dir {model_path}/pred
                    --cache_dir .cache
                    --do_predict
                    --bf16 True
                    --tf32 True
                    --max_seq_length 512
                    --per_device_eval_batch_size 16
                    --per_device_train_batch_size 4
                    --gradient_accumulation_steps 4
                    --learning_rate 2e-5
                    --num_train_epochs 40
                    --logging_strategy epoch
                    --eval_strategy epoch
                    --save_strategy epoch
                    --deepspeed {DEEPSPEED_CONFIG}
                    --overwrite_output_dir
                    --overwrite_cache
        """
        command = command.strip().split()
        print("*" * 120)
        print("[COMMAND]", " ".join(command))
        print("*" * 120)

        subprocess.run(command)
        print("\n" * 3)
