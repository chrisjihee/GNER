eval "$(conda shell.bash hook)"
conda activate GNER
cd ~/proj/GNER
rm -rf .cache_hf/datasets
bash "scripts/ZSE-yuyang-BL.sh" &> "output/ZSE-yuyang-BL-$(hostname).out"
