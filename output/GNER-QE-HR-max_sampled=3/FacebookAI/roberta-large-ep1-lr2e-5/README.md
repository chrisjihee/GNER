---
library_name: transformers
language:
- en
license: mit
base_model: FacebookAI/roberta-large
tags:
- generated_from_trainer
datasets:
- glue
metrics:
- spearmanr
model-index:
- name: roberta-large-ep1-lr2e-5
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
      value: 0.5248429103907944
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# roberta-large-ep1-lr2e-5

This model is a fine-tuned version of [FacebookAI/roberta-large](https://huggingface.co/FacebookAI/roberta-large) on the GLUE ZSE-TEST-SAMPLED-N700-QUALITY_EST-MAX_SAMPLED=0 dataset.
It achieves the following results on the evaluation set:
- Loss: 2.1793
- Mse: 2.1787
- Pearson: 0.5165
- Spearmanr: 0.5248

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
- num_epochs: 1.0

### Training results

| Training Loss | Epoch  | Step  | Validation Loss | Mse    | Pearson | Spearmanr |
|:-------------:|:------:|:-----:|:---------------:|:------:|:-------:|:---------:|
| 1.1641        | 0.0397 | 500   | 2.8400          | 2.8395 | 0.4668  | 0.4840    |
| 1.0581        | 0.0794 | 1000  | 2.3299          | 2.3294 | 0.5054  | 0.5194    |
| 0.9868        | 0.1191 | 1500  | 2.2395          | 2.2394 | 0.4966  | 0.5030    |
| 0.9425        | 0.1589 | 2000  | 2.1724          | 2.1721 | 0.4884  | 0.4913    |
| 0.9005        | 0.1986 | 2500  | 2.3620          | 2.3620 | 0.4654  | 0.4738    |
| 0.8323        | 0.2383 | 3000  | 1.6476          | 1.6475 | 0.4569  | 0.4663    |
| 0.8123        | 0.2780 | 3500  | 2.4802          | 2.4800 | 0.4872  | 0.4868    |
| 0.7891        | 0.3177 | 4000  | 2.2241          | 2.2234 | 0.4758  | 0.4795    |
| 0.7778        | 0.3574 | 4500  | 2.2665          | 2.2662 | 0.4739  | 0.4830    |
| 0.678         | 0.3971 | 5000  | 1.9307          | 1.9303 | 0.4163  | 0.4435    |
| 0.7057        | 0.4369 | 5500  | 2.0301          | 2.0299 | 0.4910  | 0.4921    |
| 0.6644        | 0.4766 | 6000  | 2.3517          | 2.3513 | 0.4111  | 0.4327    |
| 0.6212        | 0.5163 | 6500  | 1.8454          | 1.8450 | 0.4912  | 0.4980    |
| 0.5882        | 0.5560 | 7000  | 2.1257          | 2.1254 | 0.4864  | 0.4950    |
| 0.5987        | 0.5957 | 7500  | 2.1793          | 2.1787 | 0.5165  | 0.5248    |
| 0.5679        | 0.6354 | 8000  | 2.0191          | 2.0186 | 0.5094  | 0.5245    |
| 0.5656        | 0.6751 | 8500  | 2.0833          | 2.0825 | 0.4788  | 0.5001    |
| 0.5356        | 0.7149 | 9000  | 2.0446          | 2.0440 | 0.4598  | 0.4811    |
| 0.5142        | 0.7546 | 9500  | 2.2210          | 2.2208 | 0.5093  | 0.5193    |
| 0.4925        | 0.7943 | 10000 | 1.7497          | 1.7490 | 0.4209  | 0.4441    |
| 0.4822        | 0.8340 | 10500 | 2.1606          | 2.1604 | 0.4729  | 0.4807    |
| 0.4853        | 0.8737 | 11000 | 1.9533          | 1.9529 | 0.4633  | 0.4739    |
| 0.455         | 0.9134 | 11500 | 2.2088          | 2.2083 | 0.4039  | 0.4294    |
| 0.4699        | 0.9531 | 12000 | 2.1091          | 2.1088 | 0.4547  | 0.4663    |
| 0.4453        | 0.9929 | 12500 | 1.8952          | 1.8946 | 0.4607  | 0.4834    |


### Framework versions

- Transformers 4.50.0.dev0
- Pytorch 2.6.0+cu118
- Datasets 3.4.1
- Tokenizers 0.21.1
