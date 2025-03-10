---
library_name: transformers
license: mit
base_model: FacebookAI/roberta-large
tags:
- generated_from_trainer
metrics:
- spearmanr
model-index:
- name: roberta-large-num=20
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# roberta-large-num=20

This model is a fine-tuned version of [FacebookAI/roberta-large](https://huggingface.co/FacebookAI/roberta-large) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.8393
- Mse: 0.8387
- Pearson: 0.8101
- Spearmanr: 0.8216

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
| 1.3286        | 0.9994  | 820   | 1.2314          | 1.2316 | 0.6562  | 0.6672    |
| 0.8611        | 1.9994  | 1640  | 1.2152          | 1.2151 | 0.6611  | 0.6709    |
| 0.7764        | 2.9994  | 2460  | 1.2308          | 1.2309 | 0.6509  | 0.6639    |
| 0.7739        | 3.9994  | 3280  | 1.4720          | 1.4719 | 0.6305  | 0.6373    |
| 0.6286        | 4.9994  | 4100  | 1.1852          | 1.1852 | 0.6681  | 0.6739    |
| 0.5549        | 5.9994  | 4920  | 1.5497          | 1.5495 | 0.6550  | 0.6612    |
| 0.5192        | 6.9994  | 5740  | 1.1904          | 1.1903 | 0.6852  | 0.6847    |
| 0.5529        | 7.9994  | 6560  | 1.2791          | 1.2793 | 0.6640  | 0.6753    |
| 0.4123        | 8.9994  | 7380  | 0.9780          | 0.9784 | 0.7519  | 0.7642    |
| 0.3317        | 9.9994  | 8200  | 1.0093          | 1.0093 | 0.7439  | 0.7466    |
| 0.3039        | 10.9994 | 9020  | 0.9279          | 0.9280 | 0.7591  | 0.7630    |
| 0.3225        | 11.9994 | 9840  | 1.1104          | 1.1111 | 0.6971  | 0.7050    |
| 0.2376        | 12.9994 | 10660 | 0.9357          | 0.9354 | 0.7642  | 0.7804    |
| 0.1927        | 13.9994 | 11480 | 0.9300          | 0.9302 | 0.7680  | 0.7784    |
| 0.1808        | 14.9994 | 12300 | 0.9558          | 0.9555 | 0.7704  | 0.7791    |
| 0.2083        | 15.9994 | 13120 | 0.9859          | 0.9861 | 0.7692  | 0.7834    |
| 0.1436        | 16.9994 | 13940 | 0.9542          | 0.9543 | 0.7641  | 0.7767    |
| 0.1132        | 17.9994 | 14760 | 1.0253          | 1.0252 | 0.7432  | 0.7625    |
| 0.1133        | 18.9994 | 15580 | 0.9446          | 0.9447 | 0.7688  | 0.7820    |
| 0.1298        | 19.9994 | 16400 | 1.1782          | 1.1782 | 0.7245  | 0.7439    |
| 0.0903        | 20.9994 | 17220 | 0.8913          | 0.8915 | 0.7802  | 0.7943    |
| 0.0742        | 21.9994 | 18040 | 0.9530          | 0.9532 | 0.7696  | 0.7807    |
| 0.0751        | 22.9994 | 18860 | 1.0146          | 1.0143 | 0.7621  | 0.7789    |
| 0.0836        | 23.9994 | 19680 | 0.9533          | 0.9535 | 0.7947  | 0.8029    |
| 0.0616        | 24.9994 | 20500 | 1.0580          | 1.0579 | 0.7624  | 0.7772    |
| 0.055         | 25.9994 | 21320 | 0.9746          | 0.9742 | 0.7722  | 0.7913    |
| 0.0546        | 26.9994 | 22140 | 0.9935          | 0.9935 | 0.7857  | 0.7967    |
| 0.0723        | 27.9994 | 22960 | 0.9198          | 0.9199 | 0.7799  | 0.7933    |
| 0.0505        | 28.9994 | 23780 | 0.9529          | 0.9529 | 0.7817  | 0.7967    |
| 0.0538        | 29.9994 | 24600 | 0.9637          | 0.9637 | 0.7595  | 0.7771    |
| 0.0452        | 30.9994 | 25420 | 1.0722          | 1.0719 | 0.7479  | 0.7597    |
| 0.0606        | 31.9994 | 26240 | 1.0397          | 1.0391 | 0.7689  | 0.7858    |
| 0.0406        | 32.9994 | 27060 | 0.8848          | 0.8845 | 0.8143  | 0.8265    |
| 0.0371        | 33.9994 | 27880 | 0.9133          | 0.9133 | 0.8080  | 0.8228    |
| 0.0367        | 34.9994 | 28700 | 0.9399          | 0.9401 | 0.7760  | 0.7930    |
| 0.0468        | 35.9994 | 29520 | 0.8787          | 0.8788 | 0.7983  | 0.8156    |
| 0.037         | 36.9994 | 30340 | 0.9053          | 0.9055 | 0.7821  | 0.7958    |
| 0.0338        | 37.9994 | 31160 | 1.0921          | 1.0915 | 0.7631  | 0.7809    |
| 0.0349        | 38.9994 | 31980 | 0.9458          | 0.9453 | 0.8034  | 0.8179    |
| 0.0385        | 39.9994 | 32800 | 0.8393          | 0.8387 | 0.8101  | 0.8216    |


### Framework versions

- Transformers 4.50.0.dev0
- Pytorch 2.6.0+cu118
- Datasets 3.3.2
- Tokenizers 0.21.0
