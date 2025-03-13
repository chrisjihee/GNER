---
library_name: transformers
license: apache-2.0
base_model: answerdotai/ModernBERT-large
tags:
- generated_from_trainer
metrics:
- spearmanr
model-index:
- name: ModernBERT-large-num=30
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# ModernBERT-large-num=30

This model is a fine-tuned version of [answerdotai/ModernBERT-large](https://huggingface.co/answerdotai/ModernBERT-large) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.9471
- Mse: 0.9471
- Pearson: 0.7641
- Spearmanr: 0.7777

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
- gradient_accumulation_steps: 4
- total_train_batch_size: 16
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- num_epochs: 40.0

### Training results

| Training Loss | Epoch   | Step  | Validation Loss | Mse    | Pearson | Spearmanr |
|:-------------:|:-------:|:-----:|:---------------:|:------:|:-------:|:---------:|
| 3.5842        | 1.0     | 2456  | 1.1014          | 1.1013 | 0.6806  | 0.6978    |
| 2.1948        | 2.0     | 4912  | 1.3281          | 1.3282 | 0.6478  | 0.6560    |
| 1.208         | 3.0     | 7368  | 1.0391          | 1.0389 | 0.7041  | 0.7102    |
| 0.7676        | 4.0     | 9824  | 1.0710          | 1.0705 | 0.7224  | 0.7276    |
| 0.4441        | 5.0     | 12280 | 0.9340          | 0.9339 | 0.7484  | 0.7539    |
| 0.3273        | 6.0     | 14736 | 0.8961          | 0.8962 | 0.7567  | 0.7639    |
| 0.2329        | 7.0     | 17192 | 0.9732          | 0.9731 | 0.7471  | 0.7575    |
| 0.2145        | 8.0     | 19648 | 1.1420          | 1.1422 | 0.7063  | 0.7302    |
| 0.1413        | 9.0     | 22104 | 0.9862          | 0.9865 | 0.7510  | 0.7595    |
| 0.1423        | 10.0    | 24560 | 0.9471          | 0.9471 | 0.7641  | 0.7777    |
| 0.1245        | 11.0    | 27016 | 0.9741          | 0.9740 | 0.7515  | 0.7727    |
| 0.1151        | 12.0    | 29472 | 1.4486          | 1.4480 | 0.6296  | 0.6397    |
| 0.0955        | 13.0    | 31928 | 0.9576          | 0.9573 | 0.7582  | 0.7689    |
| 0.1024        | 14.0    | 34384 | 1.0280          | 1.0283 | 0.7368  | 0.7468    |
| 0.0809        | 15.0    | 36840 | 0.9888          | 0.9885 | 0.7472  | 0.7653    |
| 0.0707        | 16.0    | 39296 | 0.8930          | 0.8931 | 0.7592  | 0.7681    |
| 0.0614        | 17.0    | 41752 | 1.0002          | 1.0002 | 0.7499  | 0.7636    |
| 0.0547        | 18.0    | 44208 | 1.0984          | 1.0984 | 0.7315  | 0.7480    |
| 0.0509        | 19.0    | 46664 | 1.0744          | 1.0745 | 0.7294  | 0.7408    |
| 0.0494        | 20.0    | 49120 | 1.0090          | 1.0092 | 0.7481  | 0.7607    |
| 0.0458        | 21.0    | 51576 | 1.0620          | 1.0622 | 0.7249  | 0.7385    |
| 0.0425        | 22.0    | 54032 | 1.0912          | 1.0913 | 0.7154  | 0.7349    |
| 0.0639        | 23.0    | 56488 | 1.0745          | 1.0747 | 0.7230  | 0.7421    |
| 0.0439        | 24.0    | 58944 | 1.1740          | 1.1740 | 0.6950  | 0.7108    |
| 0.0411        | 25.0    | 61400 | 1.0859          | 1.0859 | 0.7241  | 0.7346    |
| 0.0308        | 26.0    | 63856 | 1.0776          | 1.0776 | 0.7369  | 0.7536    |
| 0.0348        | 27.0    | 66312 | 1.1755          | 1.1754 | 0.7046  | 0.7245    |
| 0.0361        | 28.0    | 68768 | 1.1559          | 1.1558 | 0.7137  | 0.7332    |
| 0.0415        | 29.0    | 71224 | 1.0429          | 1.0431 | 0.7354  | 0.7554    |
| 0.0322        | 30.0    | 73680 | 1.0360          | 1.0360 | 0.7371  | 0.7624    |
| 0.0257        | 31.0    | 76136 | 1.0493          | 1.0494 | 0.7452  | 0.7577    |
| 0.0237        | 32.0    | 78592 | 0.9891          | 0.9892 | 0.7514  | 0.7690    |
| 0.0257        | 33.0    | 81048 | 1.0016          | 1.0019 | 0.7481  | 0.7612    |
| 0.0215        | 34.0    | 83504 | 1.0295          | 1.0294 | 0.7457  | 0.7554    |
| 0.0218        | 35.0    | 85960 | 1.2234          | 1.2239 | 0.6871  | 0.7000    |
| 0.0228        | 36.0    | 88416 | 1.3102          | 1.3103 | 0.6687  | 0.6755    |
| 0.0311        | 37.0    | 90872 | 1.1584          | 1.1587 | 0.7041  | 0.7170    |
| 0.0262        | 38.0    | 93328 | 1.1187          | 1.1184 | 0.7149  | 0.7282    |
| 0.0235        | 39.0    | 95784 | 1.1168          | 1.1171 | 0.7115  | 0.7221    |
| 0.0258        | 39.9839 | 98200 | 1.1833          | 1.1837 | 0.6999  | 0.7077    |


### Framework versions

- Transformers 4.50.0.dev0
- Pytorch 2.6.0+cu118
- Datasets 3.3.2
- Tokenizers 0.21.0
