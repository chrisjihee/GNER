---
library_name: transformers
license: mit
base_model: FacebookAI/roberta-base
tags:
- generated_from_trainer
metrics:
- spearmanr
model-index:
- name: roberta-base-num=20
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# roberta-base-num=20

This model is a fine-tuned version of [FacebookAI/roberta-base](https://huggingface.co/FacebookAI/roberta-base) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 1.0893
- Mse: 1.0895
- Pearson: 0.7328
- Spearmanr: 0.7478

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
| 1.2158        | 0.9994  | 820   | 1.2336          | 1.2334 | 0.6613  | 0.6654    |
| 0.7954        | 1.9994  | 1640  | 1.3399          | 1.3398 | 0.6323  | 0.6397    |
| 0.7052        | 2.9994  | 2460  | 1.2479          | 1.2481 | 0.6655  | 0.6631    |
| 0.6872        | 3.9994  | 3280  | 1.4869          | 1.4869 | 0.6410  | 0.6430    |
| 0.5413        | 4.9994  | 4100  | 1.2181          | 1.2181 | 0.6614  | 0.6524    |
| 0.4529        | 5.9994  | 4920  | 1.6349          | 1.6352 | 0.6644  | 0.6702    |
| 0.4181        | 6.9994  | 5740  | 1.4490          | 1.4484 | 0.6668  | 0.6667    |
| 0.4332        | 7.9994  | 6560  | 1.2631          | 1.2628 | 0.6798  | 0.6857    |
| 0.3427        | 8.9994  | 7380  | 1.1724          | 1.1721 | 0.7039  | 0.7022    |
| 0.2885        | 9.9994  | 8200  | 1.0930          | 1.0932 | 0.7123  | 0.7192    |
| 0.2699        | 10.9994 | 9020  | 1.2115          | 1.2117 | 0.7084  | 0.7118    |
| 0.2795        | 11.9994 | 9840  | 1.3748          | 1.3749 | 0.6766  | 0.6831    |
| 0.222         | 12.9994 | 10660 | 1.1987          | 1.1984 | 0.6918  | 0.6992    |
| 0.1857        | 13.9994 | 11480 | 1.2358          | 1.2359 | 0.6853  | 0.7018    |
| 0.1832        | 14.9994 | 12300 | 1.1605          | 1.1604 | 0.7010  | 0.7113    |
| 0.1955        | 15.9994 | 13120 | 1.1536          | 1.1531 | 0.7003  | 0.7055    |
| 0.1512        | 16.9994 | 13940 | 1.1171          | 1.1167 | 0.7063  | 0.7161    |
| 0.1292        | 17.9994 | 14760 | 1.1316          | 1.1313 | 0.7280  | 0.7342    |
| 0.126         | 18.9994 | 15580 | 1.3131          | 1.3128 | 0.6875  | 0.7038    |
| 0.1422        | 19.9994 | 16400 | 1.1331          | 1.1327 | 0.6987  | 0.7140    |
| 0.1117        | 20.9994 | 17220 | 1.2138          | 1.2137 | 0.7006  | 0.7146    |
| 0.0928        | 21.9994 | 18040 | 1.2113          | 1.2112 | 0.6944  | 0.7123    |
| 0.0909        | 22.9994 | 18860 | 1.0713          | 1.0708 | 0.7286  | 0.7371    |
| 0.1017        | 23.9994 | 19680 | 1.1870          | 1.1867 | 0.6992  | 0.7205    |
| 0.0782        | 24.9994 | 20500 | 1.0831          | 1.0831 | 0.7312  | 0.7466    |
| 0.07          | 25.9994 | 21320 | 1.1038          | 1.1034 | 0.7173  | 0.7345    |
| 0.0677        | 26.9994 | 22140 | 1.1646          | 1.1642 | 0.7113  | 0.7282    |
| 0.0777        | 27.9994 | 22960 | 1.2142          | 1.2139 | 0.6948  | 0.7109    |
| 0.0594        | 28.9994 | 23780 | 1.1667          | 1.1663 | 0.7120  | 0.7281    |
| 0.05          | 29.9994 | 24600 | 1.0964          | 1.0963 | 0.7288  | 0.7454    |
| 0.0545        | 30.9994 | 25420 | 1.1977          | 1.1973 | 0.7260  | 0.7420    |
| 0.0632        | 31.9994 | 26240 | 1.2784          | 1.2787 | 0.6879  | 0.7081    |
| 0.0472        | 32.9994 | 27060 | 1.1544          | 1.1546 | 0.7169  | 0.7301    |
| 0.0428        | 33.9994 | 27880 | 1.0997          | 1.0991 | 0.7380  | 0.7458    |
| 0.0413        | 34.9994 | 28700 | 1.0758          | 1.0759 | 0.7345  | 0.7462    |
| 0.0479        | 35.9994 | 29520 | 1.1901          | 1.1898 | 0.7063  | 0.7221    |
| 0.0425        | 36.9994 | 30340 | 1.1323          | 1.1316 | 0.7174  | 0.7339    |
| 0.0344        | 37.9994 | 31160 | 1.1766          | 1.1760 | 0.7091  | 0.7279    |
| 0.0355        | 38.9994 | 31980 | 1.0879          | 1.0875 | 0.7340  | 0.7474    |
| 0.0443        | 39.9994 | 32800 | 1.0893          | 1.0895 | 0.7328  | 0.7478    |


### Framework versions

- Transformers 4.50.0.dev0
- Pytorch 2.6.0+cu118
- Datasets 3.3.2
- Tokenizers 0.21.0
