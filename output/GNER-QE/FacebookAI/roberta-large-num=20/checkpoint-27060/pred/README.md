---
library_name: transformers
language:
- en
base_model: output/GNER-QE/FacebookAI/roberta-large-num=20/checkpoint-27060
tags:
- generated_from_trainer
datasets:
- glue
model-index:
- name: checkpoint-27060-pred
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# checkpoint-27060-pred

This model is a fine-tuned version of [output/GNER-QE/FacebookAI/roberta-large-num=20/checkpoint-27060](https://huggingface.co/output/GNER-QE/FacebookAI/roberta-large-num=20/checkpoint-27060) on the GLUE ZSE-VALIDATION-PRED-BY_BEAM-NUM=50-VAL dataset.
It achieves the following results on the evaluation set:
- eval_loss: 0.8774
- eval_model_preparation_time: 0.5433
- eval_mse: 0.8773
- eval_pearson: 0.8110
- eval_spearmanr: 0.8229
- eval_runtime: 39.831
- eval_samples_per_second: 52.07
- eval_steps_per_second: 6.528
- step: 0

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 2e-05
- train_batch_size: 4
- eval_batch_size: 4
- seed: 42
- distributed_type: multi-GPU
- num_devices: 2
- gradient_accumulation_steps: 4
- total_train_batch_size: 32
- total_eval_batch_size: 8
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- num_epochs: 40.0

### Framework versions

- Transformers 4.50.0.dev0
- Pytorch 2.6.0+cu118
- Datasets 3.3.2
- Tokenizers 0.21.0
