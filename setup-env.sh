#!/bin/bash
# basic
conda install -n base -c conda-forge conda=25.1.1 -y;
conda create -n GNER python=3.12 -y; conda activate GNER
conda install -n GNER cuda-libraries=11.8 cuda-libraries-dev=11.8 cuda-cudart=11.8 cuda-cudart-dev=11.8 \
                      cuda-nvrtc=11.8 cuda-nvrtc-dev=11.8 cuda-driver-dev=11.8 \
                      cuda-nvcc=11.8 cuda-cccl=11.8 cuda-runtime=11.8 cuda-version=12.4 \
                      libcusparse=11 libcusparse-dev=11 libcublas=11 libcublas-dev=11 \
                      -c nvidia -c pytorch -y
pip install -U -r requirements.txt
DS_BUILD_FUSED_ADAM=1 DS_BUILD_CPU_ADAM=1 pip install --no-cache deepspeed==0.13.1

# check
conda list cuda; conda list libcu;
ds_report
huggingface-cli whoami

# option
ln -s ~/.cache/huggingface .cache_hf

# data
cd data; gzip -d -k pile-ner.json.gz; cd ..
cd data; gzip -d -k pile-ner.jsonl.gz; cd ..

# train
screen -h 1000000 -R GNER
conda activate GNER
bash scripts/train_t5_large_task_adaptation.sh
