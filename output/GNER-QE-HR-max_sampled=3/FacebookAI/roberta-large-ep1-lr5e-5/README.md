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
- name: roberta-large-ep1-lr5e-5
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
      value: 0.007387480466486327
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# roberta-large-ep1-lr5e-5

This model is a fine-tuned version of [FacebookAI/roberta-large](https://huggingface.co/FacebookAI/roberta-large) on the GLUE ZSE-TEST-SAMPLED-N700-QUALITY_EST-MAX_SAMPLED=0 dataset.
It achieves the following results on the evaluation set:
- Loss: 3.7173
- Mse: 3.7155
- Pearson: 0.0065
- Spearmanr: 0.0074

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
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
| 1.3543        | 0.0397 | 500   | 3.1607          | 3.1615 | nan     | nan       |
| 1.3429        | 0.0794 | 1000  | 1.8355          | 1.8346 | nan     | nan       |
| 1.2986        | 0.1191 | 1500  | 1.7486          | 1.7482 | nan     | nan       |
| 1.3206        | 0.1589 | 2000  | 2.0220          | 2.0209 | nan     | nan       |
| 1.3472        | 0.1986 | 2500  | 2.0924          | 2.0918 | nan     | nan       |
| 1.258         | 0.2383 | 3000  | 2.0924          | 2.0918 | nan     | nan       |
| 1.3123        | 0.2780 | 3500  | 3.2785          | 3.2765 | -0.0103 | -0.0117   |
| 1.254         | 0.3177 | 4000  | 4.0764          | 4.0687 | nan     | nan       |
| 1.252         | 0.3574 | 4500  | 5.1779          | 5.1711 | nan     | nan       |
| 1.2093        | 0.3971 | 5000  | 4.3855          | 4.3789 | 0.0063  | 0.0066    |
| 1.1983        | 0.4369 | 5500  | 4.5974          | 4.5955 | nan     | nan       |
| 1.2467        | 0.4766 | 6000  | 4.2380          | 4.2366 | -0.0114 | -0.0105   |
| 1.2236        | 0.5163 | 6500  | 3.7266          | 3.7290 | nan     | nan       |
| 1.243         | 0.5560 | 7000  | 4.4300          | 4.4323 | nan     | nan       |
| 1.2891        | 0.5957 | 7500  | 4.0179          | 4.0187 | nan     | nan       |
| 1.2222        | 0.6354 | 8000  | 4.8235          | 4.8224 | -0.0004 | -0.0003   |
| 1.2292        | 0.6751 | 8500  | 3.7173          | 3.7155 | 0.0065  | 0.0074    |
| 1.2403        | 0.7149 | 9000  | 2.8883          | 2.8902 | nan     | nan       |
| 1.2043        | 0.7546 | 9500  | 3.6477          | 3.6474 | 0.0010  | 0.0019    |
| 1.2376        | 0.7943 | 10000 | 2.8165          | 2.8171 | nan     | nan       |
| 1.2417        | 0.8340 | 10500 | 3.3266          | 3.3274 | 0.0033  | 0.0034    |
| 1.2448        | 0.8737 | 11000 | 3.5035          | 3.5010 | nan     | nan       |
| 1.2498        | 0.9134 | 11500 | 3.2042          | 3.2022 | 0.0018  | 0.0012    |
| 1.2692        | 0.9531 | 12000 | 4.3226          | 4.3260 | nan     | nan       |
| 1.2246        | 0.9929 | 12500 | 3.2368          | 3.2359 | 0.0019  | 0.0019    |


### Framework versions

- Transformers 4.50.0.dev0
- Pytorch 2.6.0+cu118
- Datasets 3.4.1
- Tokenizers 0.21.1
