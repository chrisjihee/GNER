#!/bin/bash
# basic
mamba create -n GNER python=3.11 -y
mamba activate GNER
mamba install -n GNER cuda-libraries=11.8 cuda-libraries-dev=11.8 cuda-cudart=11.8 cuda-cudart-dev=11.8 cuda-nvrtc=11.8 cuda-nvrtc-dev=11.8 cuda-driver-dev=11.8 -c nvidia
mamba install -n GNER cuda-nvcc=11.8 cuda-cccl=11.8 cuda-runtime=11.8 cuda-version=11.8 -c nvidia
mamba install -n GNER libcusparse=11 libcusparse-dev=11 libcublas=11 libcublas-dev=11 -c nvidia
#mamba install cuda-nvcc=11.8 cudatoolkit=11.8 -c nvidia -y
pip install -U -r requirements.txt
#DS_BUILD_FUSED_ADAM=1 DS_BUILD_CPU_ADAM=1 pip install --no-cache deepspeed==0.13.1
ds_report
mamba list cuda
mamba list libcu

# train
bash scripts/train_llama_task_adaptation-min4.sh
#bash scripts/train_llama_task_adaptation-min8.sh
