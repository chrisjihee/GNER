---
library_name: transformers
language:
- en
license: mit
base_model: microsoft/deberta-v3-large
tags:
- generated_from_trainer
datasets:
- glue
metrics:
- spearmanr
model-index:
- name: deberta-v3-large-ep1-lr2e-5
  results:
  - task:
      name: Text Classification
      type: text-classification
    dataset:
      name: GLUE ZSE-TEST-SAMPLED-N700-QUALITY_EST-MAX_SAMPLED=0
      type: glue
      args: ZSE-test-sampled-N700-quality_est-max_sampled=0
    metrics:
    - name: Spearmanr
      type: spearmanr
      value: 0.5515245662769198
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# deberta-v3-large-ep1-lr2e-5

This model is a fine-tuned version of [microsoft/deberta-v3-large](https://huggingface.co/microsoft/deberta-v3-large) on the GLUE ZSE-TEST-SAMPLED-N700-QUALITY_EST-MAX_SAMPLED=0 dataset.
It achieves the following results on the evaluation set:
- Loss: 1.8769
- Mse: 1.8766
- Pearson: 0.5526
- Spearmanr: 0.5515

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
- train_batch_size: 16
- eval_batch_size: 30
- seed: 42
- distributed_type: multi-GPU
- num_devices: 4
- total_train_batch_size: 64
- total_eval_batch_size: 120
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- training_steps: 7000

### Training results

| Training Loss | Epoch  | Step | Validation Loss | Mse    | Pearson | Spearmanr |
|:-------------:|:------:|:----:|:---------------:|:------:|:-------:|:---------:|
| 1.0601        | 0.0397 | 500  | 2.5313          | 2.5309 | 0.5226  | 0.5265    |
| 0.923         | 0.0794 | 1000 | 2.2387          | 2.2386 | 0.5240  | 0.5274    |
| 0.8611        | 0.1191 | 1500 | 2.4582          | 2.4580 | 0.5263  | 0.5241    |
| 0.8007        | 0.1589 | 2000 | 2.0675          | 2.0673 | 0.5240  | 0.5209    |
| 0.7227        | 0.1986 | 2500 | 2.9965          | 2.9959 | 0.5084  | 0.5123    |
| 0.6858        | 0.2383 | 3000 | 1.4963          | 1.4961 | 0.5468  | 0.5468    |
| 0.6647        | 0.2780 | 3500 | 2.1689          | 2.1686 | 0.5115  | 0.5090    |
| 0.6135        | 0.3177 | 4000 | 2.0040          | 2.0035 | 0.5431  | 0.5444    |
| 0.6085        | 0.3574 | 4500 | 2.5414          | 2.5413 | 0.5246  | 0.5246    |
| 0.4911        | 0.3971 | 5000 | 1.8061          | 1.8057 | 0.4976  | 0.5006    |
| 0.5049        | 0.4369 | 5500 | 1.8769          | 1.8766 | 0.5526  | 0.5515    |
| 0.4842        | 0.4766 | 6000 | 2.1597          | 2.1593 | 0.5350  | 0.5364    |
| 0.4215        | 0.5163 | 6500 | 1.7550          | 1.7547 | 0.4969  | 0.5066    |
| 0.4092        | 0.5560 | 7000 | 1.9364          | 1.9359 | 0.5111  | 0.5193    |


### Framework versions

- Transformers 4.50.0.dev0
- Pytorch 2.6.0+cu118
- Datasets 3.4.1
- Tokenizers 0.21.1
