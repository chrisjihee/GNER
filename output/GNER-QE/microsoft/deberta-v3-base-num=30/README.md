---
library_name: transformers
license: mit
base_model: microsoft/deberta-v3-base
tags:
- generated_from_trainer
metrics:
- spearmanr
model-index:
- name: deberta-v3-base-num=30
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# deberta-v3-base-num=30

This model is a fine-tuned version of [microsoft/deberta-v3-base](https://huggingface.co/microsoft/deberta-v3-base) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.8829
- Mse: 0.8827
- Pearson: 0.7908
- Spearmanr: 0.8065

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

| Training Loss | Epoch   | Step  | Validation Loss | Mse    | Pearson | Spearmanr |
|:-------------:|:-------:|:-----:|:---------------:|:------:|:-------:|:---------:|
| 1.0696        | 0.9994  | 1227  | 1.1548          | 1.1546 | 0.6586  | 0.6797    |
| 0.6935        | 1.9994  | 2454  | 1.3839          | 1.3837 | 0.5877  | 0.6154    |
| 0.526         | 2.9994  | 3681  | 1.1745          | 1.1743 | 0.6573  | 0.6779    |
| 0.4394        | 3.9994  | 4908  | 1.5410          | 1.5409 | 0.6551  | 0.6753    |
| 0.3465        | 4.9994  | 6135  | 1.1557          | 1.1554 | 0.6741  | 0.6947    |
| 0.298         | 5.9994  | 7362  | 1.0809          | 1.0805 | 0.7093  | 0.7258    |
| 0.2226        | 6.9994  | 8589  | 1.1391          | 1.1388 | 0.7067  | 0.7218    |
| 0.2067        | 7.9994  | 9816  | 1.2524          | 1.2524 | 0.6907  | 0.7110    |
| 0.1577        | 8.9994  | 11043 | 1.0684          | 1.0679 | 0.7170  | 0.7274    |
| 0.1352        | 9.9994  | 12270 | 1.0372          | 1.0369 | 0.7406  | 0.7493    |
| 0.1059        | 10.9994 | 13497 | 1.0747          | 1.0743 | 0.7331  | 0.7411    |
| 0.0929        | 11.9994 | 14724 | 1.0759          | 1.0754 | 0.7477  | 0.7624    |
| 0.0736        | 12.9994 | 15951 | 0.9527          | 0.9520 | 0.7533  | 0.7639    |
| 0.0634        | 13.9994 | 17178 | 1.0232          | 1.0229 | 0.7617  | 0.7714    |
| 0.0579        | 14.9994 | 18405 | 0.9840          | 0.9833 | 0.7606  | 0.7689    |
| 0.049         | 15.9994 | 19632 | 0.9436          | 0.9430 | 0.7612  | 0.7724    |
| 0.0435        | 16.9994 | 20859 | 1.1041          | 1.1038 | 0.7198  | 0.7344    |
| 0.0382        | 17.9994 | 22086 | 1.0180          | 1.0182 | 0.7496  | 0.7633    |
| 0.035         | 18.9994 | 23313 | 0.9643          | 0.9637 | 0.7515  | 0.7620    |
| 0.0309        | 19.9994 | 24540 | 0.9762          | 0.9757 | 0.7513  | 0.7657    |
| 0.0313        | 20.9994 | 25767 | 1.0188          | 1.0185 | 0.7529  | 0.7667    |
| 0.028         | 21.9994 | 26994 | 0.9538          | 0.9534 | 0.7603  | 0.7715    |
| 0.0241        | 22.9994 | 28221 | 0.9783          | 0.9780 | 0.7809  | 0.7920    |
| 0.0235        | 23.9994 | 29448 | 1.0226          | 1.0227 | 0.7559  | 0.7647    |
| 0.022         | 24.9994 | 30675 | 0.9304          | 0.9304 | 0.7706  | 0.7784    |
| 0.0198        | 25.9994 | 31902 | 0.9677          | 0.9671 | 0.7587  | 0.7698    |
| 0.0189        | 26.9994 | 33129 | 0.8829          | 0.8827 | 0.7908  | 0.8065    |
| 0.0185        | 27.9994 | 34356 | 0.8940          | 0.8933 | 0.7735  | 0.7900    |
| 0.0153        | 28.9994 | 35583 | 0.9297          | 0.9294 | 0.7677  | 0.7782    |
| 0.0153        | 29.9994 | 36810 | 0.9811          | 0.9808 | 0.7561  | 0.7673    |
| 0.0171        | 30.9994 | 38037 | 1.0237          | 1.0232 | 0.7405  | 0.7496    |
| 0.0155        | 31.9994 | 39264 | 0.9182          | 0.9175 | 0.7698  | 0.7756    |
| 0.0153        | 32.9994 | 40491 | 0.9119          | 0.9119 | 0.7768  | 0.7818    |
| 0.013         | 33.9994 | 41718 | 0.9274          | 0.9271 | 0.7669  | 0.7753    |
| 0.0134        | 34.9994 | 42945 | 0.9754          | 0.9746 | 0.7431  | 0.7623    |
| 0.0121        | 35.9994 | 44172 | 1.0059          | 1.0053 | 0.7482  | 0.7546    |
| 0.015         | 36.9994 | 45399 | 0.9600          | 0.9599 | 0.7668  | 0.7805    |
| 0.0148        | 37.9994 | 46626 | 0.9560          | 0.9555 | 0.7588  | 0.7788    |
| 0.0116        | 38.9994 | 47853 | 1.0417          | 1.0416 | 0.7459  | 0.7591    |
| 0.0132        | 39.9994 | 49080 | 0.9598          | 0.9600 | 0.7755  | 0.7921    |


### Framework versions

- Transformers 4.50.0.dev0
- Pytorch 2.6.0+cu118
- Datasets 3.3.2
- Tokenizers 0.21.0
