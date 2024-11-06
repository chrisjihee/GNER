#!/bin/bash
# conda
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
bash Miniforge3-$(uname)-$(uname -m).sh

# basic
mamba create -n GNER python=3.11 -y; mamba activate GNER
pip install -U -r requirements.txt
