eval "$(conda shell.bash hook)"
conda activate GNER
cd ~/proj/GNER
bash "scripts/ZSE-T5-Large.sh" &> "output/ZSE-T5-Large-$(hostname).log"
