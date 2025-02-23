eval "$(conda shell.bash hook)"
conda activate GNER
cd ~/proj/GNER
rm -rf .cache_hf/datasets
python "scripts/GNER-SFT-BL.py" &> "output/GNER-SFT-BL-$(hostname).out"
