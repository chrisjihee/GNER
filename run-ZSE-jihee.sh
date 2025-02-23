eval "$(conda shell.bash hook)"
conda activate GNER
cd ~/proj/GNER
rm -rf .cache_hf/datasets
python "scripts/ZSE-jihee-BL.py" &> "output/ZSE-jihee-BL-$(hostname).out"
