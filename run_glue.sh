rm -rf output/stsb
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python run_glue.py \
  --model_name_or_path google-bert/bert-base-cased \
  --task_name stsb \
  --do_train \
  --do_eval \
  --bf16 \
  --bf16_full_eval \
  --max_seq_length 256 \
  --per_device_eval_batch_size 4 \
  --per_device_train_batch_size 4 \
  --learning_rate 2e-5 \
  --num_train_epochs 3 \
  --save_strategy no \
  --eval_strategy epoch \
  --output_dir output/stsb
