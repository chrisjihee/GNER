eval "$(conda shell.bash hook)"
conda activate GNER
cd ~/proj/GNER
rm -rf .cache_hf/datasets
python "scripts/SFT-jihee-BL.py" &> "output/SFT-jihee-BL-$(hostname).out"
