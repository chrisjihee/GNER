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
- name: roberta-large-ep1-lr4e-5
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
      value: 0.00645763438341746
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# roberta-large-ep1-lr4e-5

This model is a fine-tuned version of [FacebookAI/roberta-large](https://huggingface.co/FacebookAI/roberta-large) on the GLUE ZSE-TEST-SAMPLED-N700-QUALITY_EST-MAX_SAMPLED=0 dataset.
It achieves the following results on the evaluation set:
- Loss: 2.3046
- Mse: 2.3022
- Pearson: 0.0067
- Spearmanr: 0.0065

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 4e-05
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
| 1.346         | 0.0397 | 500   | 3.0404          | 3.0407 | -0.0364 | -0.0346   |
| 1.3447        | 0.0794 | 1000  | 2.9302          | 2.9275 | 0.0051  | 0.0037    |
| 1.3122        | 0.1191 | 1500  | 2.7446          | 2.7459 | nan     | nan       |
| 1.3405        | 0.1589 | 2000  | 2.8883          | 2.8902 | nan     | nan       |
| 1.3658        | 0.1986 | 2500  | 2.9664          | 2.9653 | nan     | nan       |
| 1.3222        | 0.2383 | 3000  | 1.5280          | 1.5278 | nan     | nan       |
| 1.3507        | 0.2780 | 3500  | 2.3046          | 2.3022 | 0.0067  | 0.0065    |
| 1.3382        | 0.3177 | 4000  | 1.9538          | 1.9544 | nan     | nan       |
| 1.3469        | 0.3574 | 4500  | 2.5800          | 2.5765 | nan     | nan       |
| 1.279         | 0.3971 | 5000  | 1.8926          | 1.8923 | nan     | nan       |
| 1.2801        | 0.4369 | 5500  | 2.4513          | 2.4497 | nan     | nan       |
| 1.314         | 0.4766 | 6000  | 1.4850          | 1.4857 | -0.0033 | -0.0022   |
| 1.2886        | 0.5163 | 6500  | 1.5398          | 1.5423 | -0.0003 | 0.0007    |
| 1.2843        | 0.5560 | 7000  | 1.5625          | 1.5622 | nan     | nan       |
| 1.3371        | 0.5957 | 7500  | 1.4954          | 1.4957 | nan     | nan       |
| 1.2852        | 0.6354 | 8000  | 1.5721          | 1.5712 | 0.0012  | 0.0007    |
| 1.2845        | 0.6751 | 8500  | 2.5429          | 2.5440 | nan     | nan       |
| 1.2802        | 0.7149 | 9000  | 1.5093          | 1.5090 | -0.0017 | -0.0004   |
| 1.2339        | 0.7546 | 9500  | 1.4969          | 1.4964 | -0.0060 | -0.0055   |
| 1.2752        | 0.7943 | 10000 | 1.5052          | 1.5056 | nan     | nan       |
| 1.2757        | 0.8340 | 10500 | 1.5637          | 1.5666 | -0.0018 | -0.0020   |
| 1.2746        | 0.8737 | 11000 | 1.5381          | 1.5362 | nan     | nan       |
| 1.2715        | 0.9134 | 11500 | 1.4890          | 1.4882 | nan     | nan       |
| 1.3274        | 0.9531 | 12000 | 1.6742          | 1.6740 | nan     | nan       |
| 1.2485        | 0.9929 | 12500 | 1.5323          | 1.5312 | 0.0001  | 0.0013    |


### Framework versions

- Transformers 4.50.0.dev0
- Pytorch 2.6.0+cu118
- Datasets 3.4.1
- Tokenizers 0.21.1
