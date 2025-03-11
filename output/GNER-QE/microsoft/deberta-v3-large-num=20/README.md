---
library_name: transformers
license: mit
base_model: microsoft/deberta-v3-large
tags:
- generated_from_trainer
metrics:
- spearmanr
model-index:
- name: deberta-v3-large-num=20
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# deberta-v3-large-num=20

This model is a fine-tuned version of [microsoft/deberta-v3-large](https://huggingface.co/microsoft/deberta-v3-large) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.7406
- Mse: 0.7405
- Pearson: 0.8328
- Spearmanr: 0.8455

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
| 1.0242        | 0.9994  | 820   | 1.0003          | 1.0006 | 0.7529  | 0.7698    |
| 0.4883        | 1.9994  | 1640  | 0.9629          | 0.9627 | 0.7933  | 0.7976    |
| 0.3627        | 2.9994  | 2460  | 0.9591          | 0.9589 | 0.7815  | 0.7939    |
| 0.3212        | 3.9994  | 3280  | 0.8707          | 0.8705 | 0.7883  | 0.8005    |
| 0.192         | 4.9994  | 4100  | 0.8395          | 0.8397 | 0.8187  | 0.8306    |
| 0.115         | 5.9994  | 4920  | 0.7401          | 0.7397 | 0.8205  | 0.8323    |
| 0.1012        | 6.9994  | 5740  | 0.7253          | 0.7248 | 0.8327  | 0.8428    |
| 0.0974        | 7.9994  | 6560  | 0.8074          | 0.8073 | 0.8158  | 0.8244    |
| 0.078         | 8.9994  | 7380  | 0.8036          | 0.8033 | 0.8190  | 0.8296    |
| 0.0562        | 9.9994  | 8200  | 0.6410          | 0.6409 | 0.8523  | 0.8605    |
| 0.0539        | 10.9994 | 9020  | 0.7156          | 0.7154 | 0.8358  | 0.8438    |
| 0.0577        | 11.9994 | 9840  | 0.7661          | 0.7659 | 0.8284  | 0.8349    |
| 0.0498        | 12.9994 | 10660 | 0.7841          | 0.7838 | 0.8338  | 0.8449    |
| 0.0337        | 13.9994 | 11480 | 0.7220          | 0.7216 | 0.8312  | 0.8420    |
| 0.0315        | 14.9994 | 12300 | 0.8036          | 0.8034 | 0.8307  | 0.8414    |
| 0.0412        | 15.9994 | 13120 | 0.8791          | 0.8791 | 0.8057  | 0.8141    |
| 0.0387        | 16.9994 | 13940 | 0.7457          | 0.7453 | 0.8387  | 0.8469    |
| 0.0238        | 17.9994 | 14760 | 0.6586          | 0.6587 | 0.8431  | 0.8575    |
| 0.0241        | 18.9994 | 15580 | 0.8449          | 0.8442 | 0.8133  | 0.8200    |
| 0.0323        | 19.9994 | 16400 | 0.7235          | 0.7231 | 0.8318  | 0.8346    |
| 0.0269        | 20.9994 | 17220 | 0.7969          | 0.7961 | 0.8192  | 0.8270    |
| 0.0236        | 21.9994 | 18040 | 0.7194          | 0.7189 | 0.8449  | 0.8527    |
| 0.0279        | 22.9994 | 18860 | 0.8143          | 0.8138 | 0.8270  | 0.8397    |
| 0.0266        | 23.9994 | 19680 | 0.7676          | 0.7674 | 0.8249  | 0.8363    |
| 0.0223        | 24.9994 | 20500 | 0.8339          | 0.8340 | 0.8134  | 0.8253    |
| 0.0165        | 25.9994 | 21320 | 0.8477          | 0.8475 | 0.8147  | 0.8302    |
| 0.0186        | 26.9994 | 22140 | 0.7824          | 0.7823 | 0.8321  | 0.8432    |
| 0.0212        | 27.9994 | 22960 | 0.7358          | 0.7355 | 0.8435  | 0.8553    |
| 0.02          | 28.9994 | 23780 | 0.8847          | 0.8841 | 0.8117  | 0.8254    |
| 0.0112        | 29.9994 | 24600 | 0.6946          | 0.6942 | 0.8399  | 0.8517    |
| 0.0149        | 30.9994 | 25420 | 0.7901          | 0.7897 | 0.8216  | 0.8363    |
| 0.0211        | 31.9994 | 26240 | 0.8924          | 0.8921 | 0.8021  | 0.8176    |
| 0.0162        | 32.9994 | 27060 | 0.7287          | 0.7288 | 0.8371  | 0.8492    |
| 0.0109        | 33.9994 | 27880 | 0.7265          | 0.7266 | 0.8296  | 0.8451    |
| 0.0144        | 34.9994 | 28700 | 0.7412          | 0.7405 | 0.8350  | 0.8494    |
| 0.0144        | 35.9994 | 29520 | 0.7923          | 0.7922 | 0.8233  | 0.8377    |
| 0.0129        | 36.9994 | 30340 | 0.9330          | 0.9330 | 0.7988  | 0.8159    |
| 0.0192        | 37.9994 | 31160 | 0.8362          | 0.8358 | 0.8252  | 0.8394    |
| 0.0139        | 38.9994 | 31980 | 0.7494          | 0.7493 | 0.8406  | 0.8569    |
| 0.0136        | 39.9994 | 32800 | 0.7406          | 0.7405 | 0.8328  | 0.8455    |


### Framework versions

- Transformers 4.50.0.dev0
- Pytorch 2.6.0+cu118
- Datasets 3.3.2
- Tokenizers 0.21.0
