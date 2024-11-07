#!/bin/bash
# basic
mamba create -n GNER python=3.11 -y
mamba activate GNER
mamba install -n GNER cuda-libraries-dev=11.8 cuda-libraries=11.8 cuda-nvcc=11.8 cuda-nvrtc=11.8 cuda-nvrtc-dev=11.8 cuda-runtime=11.8 cuda-cccl=11.8 -c nvidia -c pytorch -y
#mamba install cuda-nvcc=11.8 cudatoolkit=11.8 -c nvidia -y
pip install -U -r requirements.txt

# train
#bash scripts/train_llama_task_adaptation-min4.sh
bash scripts/train_llama_task_adaptation-min8.sh
