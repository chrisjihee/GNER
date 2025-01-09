#!/bin/bash
# basic
mamba create -n GNER python=3.11 -y
mamba activate GNER
mamba install -n GNER cuda-libraries=11.8 cuda-libraries-dev=11.8 cuda-cudart=11.8 cuda-cudart-dev=11.8 \
                      cuda-nvrtc=11.8 cuda-nvrtc-dev=11.8 cuda-driver-dev=11.8 \
                      cuda-nvcc=11.8 cuda-cccl=11.8 cuda-runtime=11.8 cuda-version=12.4 \
                      libcusparse=11 libcusparse-dev=11 libcublas=11 libcublas-dev=11 \
                      -c nvidia -c pytorch -y
pip install -U -r requirements.txt
pip uninstall deepspeed
DS_BUILD_FUSED_ADAM=1 pip install --no-cache deepspeed==0.13.1  # [OK]
DS_BUILD_FUSED_ADAM=1 DS_BUILD_CPU_ADAM=1 pip install --no-cache deepspeed==0.13.1  # [OK]

# check
ds_report
mamba list cuda; mamba list libcu;

# option
#ln -s ~/.cache/huggingface .cache_hf

# train
#bash scripts/train_llama_task_adaptation-min4.sh
#bash scripts/train_llama_task_adaptation-min8.sh
bash scripts/train_llama_task_adaptation-full.sh
