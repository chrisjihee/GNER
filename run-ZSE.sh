eval "$(conda shell.bash hook)"
conda activate GNER
cd ~/proj/GNER
rm -rf .cache_hf/datasets
bash "scripts/ZSE-T5-Large.sh" &> "output/ZSE-T5-Large-$(hostname).out"
