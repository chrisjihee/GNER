eval "$(conda shell.bash hook)"
conda activate GNER
cd ~/proj/GNER
bash "scripts/SFT-T5-Large.sh" &> "output/SFT-T5-Large-$(hostname).out"
