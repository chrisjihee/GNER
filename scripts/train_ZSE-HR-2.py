import os
import random
import socket
import subprocess

from base import *

# Environment variables
debugging = False
port = random.randint(25000, 30000)
hostname = socket.gethostname()
cuda_devices = os.getenv("CUDA_VISIBLE_DEVICES", "4,5" if not debugging else "0")
source_file = "train_GNER.py"

# Training arguments
experiment_type = "HR-2"
output_name = f"train_ZSE-{experiment_type}-{hostname}"
train_file = "data/pile-ner-sampled-N30129-HR298291,103814,194477.jsonl"
eval_file = "data/ZSE-validation-sampled-N210-HR3100,700,2400.jsonl"
pred_file = "data/ZSE-test-sampled-N700-HR10100,2100,8000.jsonl"
metric_for_best_model = "eval_average"
max_generation_tokens = 640
save_total_limit = 3
train_epochs = 12
eval_epochs = 0.5
save_epochs = eval_epochs
learning_rate = 5e-5
logging_steps = 10
gradient_steps = 4
train_batch = 8
eval_batch = 8

# Loop through each model and dataset
for ds_config, run_prefix, pretrained in model_specs:
    command = "rm -rf .cache_hf/datasets".strip().split()
    print("*" * 120)
    print("[COMMAND]", " ".join(command))
    print("*" * 120)
    subprocess.run(command)
    print("\n" * 3)

    suffix = f"-{experiment_type}"
    run_version = f"{run_prefix}{suffix}"
    use_flash_attention = pretrained.startswith("microsoft/Phi")

    command = f"""
        python -m
            deepspeed.launcher.runner
                --include=localhost:{cuda_devices}
                --master_port={port}
            {source_file}
                --pred_file {pred_file}
                --eval_file {eval_file}
                --train_file {train_file}
                --output_file train-metrics-{train_epochs}ep.csv
                --logging_file train-loggings-{train_epochs}ep.out
                --output_name {output_name}
                --run_version {run_version}
                --pretrained {pretrained}
                --trainer_deepspeed {ds_config}
                --num_train_epochs {train_epochs}
                --eval_epochs {eval_epochs}
                --save_epochs {save_epochs}
                --learning_rate {learning_rate}
                --logging_steps {logging_steps}
                --per_device_eval_batch_size {eval_batch}
                --per_device_train_batch_size {train_batch}
                --gradient_accumulation_steps {gradient_steps}
                --max_generation_tokens {max_generation_tokens}
                --save_total_limit {save_total_limit}
                --metric_for_best_model {metric_for_best_model}
                --{'' if use_flash_attention else 'no_'}use_flash_attention
                --{'' if debugging else 'no_'}debugging
    """
    command = command.strip().split()
    print("*" * 120)
    print("[COMMAND]", " ".join(command))
    print("*" * 120)

    subprocess.run(command)
    print("\n" * 3)
