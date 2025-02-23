eval "$(conda shell.bash hook)"
conda activate GNER
cd ~/proj/GNER
rm -rf .cache_hf/datasets
bash "scripts/SFT-yuyang-BL.sh" &> "output/SFT-yuyang-BL-$(hostname).out"
