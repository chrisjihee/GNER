---
library_name: transformers
license: apache-2.0
base_model: google-bert/bert-base-cased
tags:
- generated_from_trainer
metrics:
- spearmanr
model-index:
- name: bert-base-cased-num=10
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# bert-base-cased-num=10

This model is a fine-tuned version of [google-bert/bert-base-cased](https://huggingface.co/google-bert/bert-base-cased) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 1.1623
- Mse: 1.1626
- Pearson: 0.7158
- Spearmanr: 0.7255

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

### Training results

| Training Loss | Epoch | Step  | Validation Loss | Mse    | Pearson | Spearmanr |
|:-------------:|:-----:|:-----:|:---------------:|:------:|:-------:|:---------:|
| 1.5176        | 1.0   | 412   | 1.4303          | 1.4310 | 0.5930  | 0.6021    |
| 0.8894        | 2.0   | 824   | 1.3635          | 1.3634 | 0.6339  | 0.6424    |
| 0.7179        | 3.0   | 1236  | 1.4371          | 1.4375 | 0.6418  | 0.6646    |
| 0.6174        | 4.0   | 1648  | 1.2820          | 1.2816 | 0.6506  | 0.6447    |
| 0.532         | 5.0   | 2060  | 1.2700          | 1.2693 | 0.6512  | 0.6428    |
| 0.4651        | 6.0   | 2472  | 1.3316          | 1.3316 | 0.6416  | 0.6325    |
| 0.4189        | 7.0   | 2884  | 1.2135          | 1.2116 | 0.6840  | 0.6790    |
| 0.3586        | 8.0   | 3296  | 1.2672          | 1.2665 | 0.6795  | 0.6818    |
| 0.3282        | 9.0   | 3708  | 1.5041          | 1.5047 | 0.6319  | 0.6318    |
| 0.2819        | 10.0  | 4120  | 1.2867          | 1.2870 | 0.6516  | 0.6484    |
| 0.2503        | 11.0  | 4532  | 1.4824          | 1.4824 | 0.6755  | 0.6760    |
| 0.22          | 12.0  | 4944  | 1.2807          | 1.2818 | 0.6777  | 0.6845    |
| 0.1948        | 13.0  | 5356  | 1.1654          | 1.1641 | 0.7078  | 0.7129    |
| 0.1753        | 14.0  | 5768  | 1.2266          | 1.2239 | 0.7015  | 0.7054    |
| 0.1559        | 15.0  | 6180  | 1.2200          | 1.2192 | 0.7032  | 0.7045    |
| 0.1415        | 16.0  | 6592  | 1.2307          | 1.2312 | 0.6954  | 0.6965    |
| 0.1263        | 17.0  | 7004  | 1.2884          | 1.2885 | 0.6919  | 0.6950    |
| 0.1135        | 18.0  | 7416  | 1.2209          | 1.2212 | 0.7108  | 0.7161    |
| 0.1031        | 19.0  | 7828  | 1.1577          | 1.1559 | 0.6964  | 0.7002    |
| 0.0927        | 20.0  | 8240  | 1.1606          | 1.1603 | 0.7161  | 0.7265    |
| 0.0869        | 21.0  | 8652  | 1.2406          | 1.2413 | 0.7060  | 0.7164    |
| 0.0816        | 22.0  | 9064  | 1.2902          | 1.2899 | 0.6886  | 0.6929    |
| 0.0737        | 23.0  | 9476  | 1.2149          | 1.2154 | 0.7139  | 0.7163    |
| 0.068         | 24.0  | 9888  | 1.2743          | 1.2739 | 0.7014  | 0.7071    |
| 0.0624        | 25.0  | 10300 | 1.3351          | 1.3343 | 0.6733  | 0.6879    |
| 0.059         | 26.0  | 10712 | 1.2586          | 1.2576 | 0.6816  | 0.6877    |
| 0.0549        | 27.0  | 11124 | 1.2508          | 1.2505 | 0.7023  | 0.7074    |
| 0.0549        | 28.0  | 11536 | 1.1782          | 1.1790 | 0.6972  | 0.7062    |
| 0.0497        | 29.0  | 11948 | 1.1881          | 1.1876 | 0.7089  | 0.7169    |
| 0.0461        | 30.0  | 12360 | 1.0632          | 1.0638 | 0.7494  | 0.7591    |
| 0.0435        | 31.0  | 12772 | 1.2070          | 1.2076 | 0.6973  | 0.7037    |
| 0.0424        | 32.0  | 13184 | 1.1306          | 1.1316 | 0.7171  | 0.7203    |
| 0.0396        | 33.0  | 13596 | 1.2602          | 1.2615 | 0.6988  | 0.7052    |
| 0.0384        | 34.0  | 14008 | 1.1200          | 1.1203 | 0.7302  | 0.7358    |
| 0.0369        | 35.0  | 14420 | 1.1543          | 1.1544 | 0.7117  | 0.7196    |
| 0.0336        | 36.0  | 14832 | 1.1698          | 1.1694 | 0.7119  | 0.7227    |
| 0.0345        | 37.0  | 15244 | 1.1582          | 1.1587 | 0.7243  | 0.7288    |
| 0.0311        | 38.0  | 15656 | 1.1538          | 1.1536 | 0.7194  | 0.7264    |
| 0.0286        | 39.0  | 16068 | 1.1514          | 1.1514 | 0.7189  | 0.7295    |
| 0.029         | 40.0  | 16480 | 1.1623          | 1.1626 | 0.7158  | 0.7255    |


### Framework versions

- Transformers 4.50.0.dev0
- Pytorch 2.6.0+cu118
- Datasets 3.3.2
- Tokenizers 0.21.0
