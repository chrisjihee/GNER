eval "$(conda shell.bash hook)"
conda activate GNER
cd ~/proj/GNER
rm -rf .cache_hf/datasets
bash "scripts/SFT-yuyang-T5-Large.sh" &> "output/SFT-yuyang-T5-Large-$(hostname).out"
