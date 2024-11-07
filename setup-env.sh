#!/bin/bash
# basic
mamba create -n GNER python=3.11 -y
mamba activate GNER
mamba install cuda-nvcc=11.8 cudatoolkit=11.8 -c nvidia -y
pip install -U -r requirements.txt

# train
bash scripts/train_llama_task_adaptation-min4.sh
#bash scripts/train_llama_task_adaptation-min8.sh
