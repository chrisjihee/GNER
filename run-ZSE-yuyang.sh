eval "$(conda shell.bash hook)"
conda activate GNER
cd ~/proj/GNER
rm -rf .cache_hf/datasets
bash "scripts/ZSE-yuyang-T5-Large.sh" &> "output/ZSE-yuyang-T5-Large-$(hostname).out"
