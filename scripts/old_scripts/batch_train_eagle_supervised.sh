cd /raid/chrisjihee/proj/GNER

echo ""
echo ""
echo "========================================================================================================================"
echo "[bash scripts/train_eagle_1b_supervised-plus-plus-plus.sh]"
echo "========================================================================================================================"
bash scripts/train_eagle_1b_supervised-plus-plus-plus.sh
echo "========================================================================================================================"
echo ""

echo ""
echo ""
echo "========================================================================================================================"
echo "[bash scripts/train_eagle_1b_supervised-plus-plus.sh]"
echo "========================================================================================================================"
bash scripts/train_eagle_1b_supervised-plus-plus.sh
echo "========================================================================================================================"
echo ""

echo ""
echo ""
echo "========================================================================================================================"
echo "[bash scripts/train_eagle_1b_supervised-plus.sh]"
echo "========================================================================================================================"
bash scripts/train_eagle_1b_supervised-plus.sh
echo "========================================================================================================================"
echo ""

echo ""
echo ""
echo "========================================================================================================================"
echo "[bash scripts/train_eagle_1b_supervised-base.sh]"
echo "========================================================================================================================"
bash scripts/train_eagle_1b_supervised-base.sh
echo "========================================================================================================================"
echo ""

echo ""
echo ""
echo "========================================================================================================================"
echo "[bash scripts/train_eagle_3b_supervised-plus-plus-plus.sh]"
echo "========================================================================================================================"
bash scripts/train_eagle_3b_supervised-plus-plus-plus.sh
echo "========================================================================================================================"
echo ""

echo ""
echo ""
echo "========================================================================================================================"
echo "[bash scripts/train_eagle_3b_supervised-plus-plus.sh]"
echo "========================================================================================================================"
bash scripts/train_eagle_3b_supervised-plus-plus.sh
echo "========================================================================================================================"
echo ""

echo ""
echo ""
echo "========================================================================================================================"
echo "[bash scripts/train_eagle_3b_supervised-plus.sh]"
echo "========================================================================================================================"
bash scripts/train_eagle_3b_supervised-plus.sh
echo "========================================================================================================================"
echo ""

echo ""
echo ""
echo "========================================================================================================================"
echo "[bash scripts/train_eagle_3b_supervised-base.sh]"
echo "========================================================================================================================"
bash scripts/train_eagle_3b_supervised-base.sh
echo "========================================================================================================================"
echo ""

echo ""
echo ""
echo "************************************************************************************************************************"
echo "[bash scripts/batch_train_llama3_supervised.sh]"
echo "************************************************************************************************************************"
bash scripts/batch_train_llama3_supervised.sh
echo "************************************************************************************************************************"
echo ""
