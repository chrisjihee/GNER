---
library_name: transformers
license: mit
base_model: FacebookAI/roberta-large
tags:
- generated_from_trainer
metrics:
- spearmanr
model-index:
- name: roberta-large-num=30
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# roberta-large-num=30

This model is a fine-tuned version of [FacebookAI/roberta-large](https://huggingface.co/FacebookAI/roberta-large) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.9598
- Mse: 0.9595
- Pearson: 0.7738
- Spearmanr: 0.7926

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
| 1.1606        | 0.9994  | 1227  | 1.1432          | 1.1440 | 0.6995  | 0.7141    |
| 0.7531        | 1.9994  | 2454  | 1.0759          | 1.0751 | 0.6977  | 0.7024    |
| 0.5129        | 2.9994  | 3681  | 0.8535          | 0.8534 | 0.7617  | 0.7691    |
| 0.4114        | 3.9994  | 4908  | 1.0947          | 1.0944 | 0.7558  | 0.7605    |
| 0.2961        | 4.9994  | 6135  | 0.8099          | 0.8096 | 0.7772  | 0.7876    |
| 0.2439        | 5.9994  | 7362  | 0.7638          | 0.7638 | 0.8053  | 0.8220    |
| 0.1702        | 6.9994  | 8589  | 0.7751          | 0.7751 | 0.8162  | 0.8292    |
| 0.1532        | 7.9994  | 9816  | 0.7476          | 0.7475 | 0.8112  | 0.8281    |
| 0.109         | 8.9994  | 11043 | 0.7314          | 0.7314 | 0.8173  | 0.8279    |
| 0.104         | 9.9994  | 12270 | 0.8818          | 0.8817 | 0.7829  | 0.8000    |
| 0.0824        | 10.9994 | 13497 | 0.7520          | 0.7519 | 0.8223  | 0.8353    |
| 0.0826        | 11.9994 | 14724 | 0.9057          | 0.9060 | 0.8002  | 0.8103    |
| 0.0636        | 12.9994 | 15951 | 0.7676          | 0.7672 | 0.8122  | 0.8228    |
| 0.0624        | 13.9994 | 17178 | 0.8921          | 0.8924 | 0.7944  | 0.8055    |
| 0.0534        | 14.9994 | 18405 | 0.8337          | 0.8335 | 0.7985  | 0.8113    |
| 0.0514        | 15.9994 | 19632 | 0.8387          | 0.8385 | 0.7939  | 0.8037    |
| 0.0461        | 16.9994 | 20859 | 0.8598          | 0.8599 | 0.7988  | 0.8139    |
| 0.0488        | 17.9994 | 22086 | 0.8315          | 0.8314 | 0.8004  | 0.8151    |
| 0.0412        | 18.9994 | 23313 | 0.7821          | 0.7824 | 0.8093  | 0.8240    |
| 0.0426        | 19.9994 | 24540 | 0.8227          | 0.8226 | 0.8019  | 0.8131    |
| 0.0353        | 20.9994 | 25767 | 0.9079          | 0.9076 | 0.7900  | 0.8040    |
| 0.0341        | 21.9994 | 26994 | 0.8162          | 0.8161 | 0.8046  | 0.8161    |
| 0.0324        | 22.9994 | 28221 | 0.8269          | 0.8269 | 0.8010  | 0.8132    |
| 0.0289        | 23.9994 | 29448 | 0.8789          | 0.8788 | 0.7846  | 0.8000    |
| 0.0249        | 24.9994 | 30675 | 0.8881          | 0.8880 | 0.7872  | 0.7977    |
| 0.0285        | 25.9994 | 31902 | 0.8908          | 0.8912 | 0.7953  | 0.8095    |
| 0.0234        | 26.9994 | 33129 | 0.8258          | 0.8258 | 0.8025  | 0.8175    |
| 0.0222        | 27.9994 | 34356 | 0.9851          | 0.9847 | 0.7657  | 0.7826    |
| 0.0215        | 28.9994 | 35583 | 0.8242          | 0.8244 | 0.8041  | 0.8150    |
| 0.0231        | 29.9994 | 36810 | 1.0407          | 1.0407 | 0.7678  | 0.7814    |
| 0.0191        | 30.9994 | 38037 | 0.8830          | 0.8828 | 0.7965  | 0.8145    |
| 0.0199        | 31.9994 | 39264 | 0.8727          | 0.8729 | 0.7811  | 0.7967    |
| 0.0176        | 32.9994 | 40491 | 0.8455          | 0.8450 | 0.7977  | 0.8109    |
| 0.0225        | 33.9994 | 41718 | 0.7015          | 0.7014 | 0.8266  | 0.8377    |
| 0.0159        | 34.9994 | 42945 | 0.8061          | 0.8060 | 0.7997  | 0.8147    |
| 0.014         | 35.9994 | 44172 | 0.8028          | 0.8028 | 0.8117  | 0.8203    |
| 0.0131        | 36.9994 | 45399 | 0.9389          | 0.9390 | 0.7778  | 0.7947    |
| 0.0147        | 37.9994 | 46626 | 0.8526          | 0.8530 | 0.7960  | 0.8127    |
| 0.0134        | 38.9994 | 47853 | 0.8550          | 0.8550 | 0.7990  | 0.8127    |
| 0.0142        | 39.9994 | 49080 | 0.9598          | 0.9595 | 0.7738  | 0.7926    |


### Framework versions

- Transformers 4.50.0.dev0
- Pytorch 2.6.0+cu118
- Datasets 3.3.2
- Tokenizers 0.21.0
