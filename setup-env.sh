#!/bin/bash
# 1. Install Miniforge
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh

# 2. Clone the repository
rm -rf GNER*; git clone https://github.com/chrisjihee/GNER.git; cd GNER*;

# 3. Create a new environment
conda search conda -c conda-forge
conda install -n base -c conda-forge conda=25.1.1 -y
conda create -n GNER python=3.12 -y
conda activate GNER
conda install -n GNER cuda-libraries=11.8 cuda-libraries-dev=11.8 cudatoolkit=11.8 cuda-cudart=11.8 cuda-cudart-dev=11.8 cuda-nvrtc=11.8 cuda-nvrtc-dev=11.8 cuda-nvcc=11.8 cuda-cccl=11.8 cuda-runtime=11.8 libcusparse=11 libcusparse-dev=11 libcublas=11 libcublas-dev=11 -c nvidia -c pytorch -y
conda list > cuda_versions.txt

# 4. Install the required packages
pip install -r requirements.txt; pip list | grep torch
rm -rf transformers; git clone https://github.com/chrisjihee/transformers.git; pip install -U -e transformers
rm -rf chrisbase;    git clone https://github.com/chrisjihee/chrisbase.git;    pip install -U -e chrisbase
rm -rf chrisdata;    git clone https://github.com/chrisjihee/chrisdata.git;    pip install -U -e chrisdata
rm -rf progiter;     git clone https://github.com/chrisjihee/progiter.git;     pip install -U -e progiter
export CUDA_HOME=""; DS_BUILD_FUSED_ADAM=1 pip install --no-cache deepspeed; ds_report
MAX_JOBS=40 pip install --no-cache --no-build-isolation --upgrade flash-attn;  # for Micorsoft's models
pip list | grep -E "torch|transformer|lightning |accelerate|deepspeed|flash_attn|numpy|sentencepiece|eval|chris|prog|pydantic" > dep_versions.txt

# 5. Unzip some archived data
cd data; gzip -d -k pile-ner.jsonl.gz; cd ..
python chrisdata/scripts/ner_sample_jsonl.py
mkdir -p output; mkdir -p output-lfs
ln -s /dlfs/jhryu/GNER-output-dl012 output-lfs-dl012  # for dl012
ln -s /dlfs/jhryu/GNER-output-dl026 output-lfs-dl026  # for dl026

# 6. Login to Hugging Face and link the cache
huggingface-cli whoami
huggingface-cli login
ln -s ~/.cache/huggingface ./.cache_hf
ln -s ~/.cache/huggingface/hub ./.cache

# 7. Run the training script
screen -dmS GNER bash -c ~/proj/GNER/run-SFT-jihee.sh; screen -ls; sleep 3s; tail -f $(ls -t ~/proj/GNER/output/*.out | head -n 1)
screen -dmS GNER bash -c ~/proj/GNER/run-SFT-yuyang.sh; screen -ls; sleep 3s; tail -f $(ls -t ~/proj/GNER/output/*.out | head -n 1)
screen -dmS GNER bash -c ~/proj/GNER/run-ZSE-jihee.sh; screen -ls; sleep 3s; tail -f $(ls -t ~/proj/GNER/output/*.out | head -n 1)
screen -dmS GNER bash -c ~/proj/GNER/run-ZSE-yuyang.sh; screen -ls; sleep 3s; tail -f $(ls -t ~/proj/GNER/output/*.out | head -n 1)
