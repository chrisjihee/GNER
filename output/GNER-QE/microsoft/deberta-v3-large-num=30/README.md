---
library_name: transformers
license: mit
base_model: microsoft/deberta-v3-large
tags:
- generated_from_trainer
metrics:
- spearmanr
model-index:
- name: deberta-v3-large-num=30
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# deberta-v3-large-num=30

This model is a fine-tuned version of [microsoft/deberta-v3-large](https://huggingface.co/microsoft/deberta-v3-large) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.7377
- Mse: 0.7378
- Pearson: 0.8345
- Spearmanr: 0.8448

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
| 0.8126        | 0.9994  | 1227  | 0.8938          | 0.8938 | 0.7558  | 0.7666    |
| 0.4239        | 1.9994  | 2454  | 0.7562          | 0.7562 | 0.8057  | 0.8150    |
| 0.2206        | 2.9994  | 3681  | 0.8407          | 0.8406 | 0.7955  | 0.8027    |
| 0.1338        | 3.9994  | 4908  | 1.0619          | 1.0619 | 0.7531  | 0.7681    |
| 0.0884        | 4.9994  | 6135  | 0.9028          | 0.9028 | 0.7862  | 0.8050    |
| 0.07          | 5.9994  | 7362  | 1.0659          | 1.0659 | 0.7386  | 0.7502    |
| 0.0524        | 6.9994  | 8589  | 0.9560          | 0.9560 | 0.7775  | 0.7911    |
| 0.0429        | 7.9994  | 9816  | 0.9599          | 0.9597 | 0.7627  | 0.7799    |
| 0.0365        | 8.9994  | 11043 | 1.0044          | 1.0042 | 0.7542  | 0.7740    |
| 0.0406        | 9.9994  | 12270 | 0.9383          | 0.9383 | 0.7773  | 0.7924    |
| 0.0326        | 10.9994 | 13497 | 1.0471          | 1.0471 | 0.7744  | 0.7923    |
| 0.0321        | 11.9994 | 14724 | 0.8653          | 0.8648 | 0.8007  | 0.8117    |
| 0.0227        | 12.9994 | 15951 | 1.1360          | 1.1361 | 0.7305  | 0.7510    |
| 0.023         | 13.9994 | 17178 | 0.9707          | 0.9709 | 0.7573  | 0.7835    |
| 0.0225        | 14.9994 | 18405 | 1.0990          | 1.0989 | 0.7503  | 0.7784    |
| 0.0235        | 15.9994 | 19632 | 0.8754          | 0.8757 | 0.7830  | 0.7999    |
| 0.0213        | 16.9994 | 20859 | 1.0197          | 1.0198 | 0.7672  | 0.7881    |
| 0.017         | 17.9994 | 22086 | 0.8623          | 0.8622 | 0.7990  | 0.8147    |
| 0.0188        | 18.9994 | 23313 | 0.8944          | 0.8938 | 0.8024  | 0.8203    |
| 0.0149        | 19.9994 | 24540 | 0.9560          | 0.9559 | 0.7747  | 0.7914    |
| 0.0185        | 20.9994 | 25767 | 0.9107          | 0.9110 | 0.7909  | 0.8090    |
| 0.0144        | 21.9994 | 26994 | 0.8934          | 0.8934 | 0.7975  | 0.8138    |
| 0.0156        | 22.9994 | 28221 | 0.8761          | 0.8761 | 0.8019  | 0.8141    |
| 0.0154        | 23.9994 | 29448 | 0.8279          | 0.8271 | 0.8124  | 0.8229    |
| 0.012         | 24.9994 | 30675 | 0.8130          | 0.8123 | 0.8121  | 0.8271    |
| 0.0126        | 25.9994 | 31902 | 0.8543          | 0.8545 | 0.7941  | 0.8116    |
| 0.0146        | 26.9994 | 33129 | 0.7993          | 0.7991 | 0.8152  | 0.8321    |
| 0.0163        | 27.9994 | 34356 | 0.9094          | 0.9092 | 0.7944  | 0.8134    |
| 0.0115        | 28.9994 | 35583 | 0.7752          | 0.7749 | 0.8183  | 0.8344    |
| 0.0089        | 29.9994 | 36810 | 0.7920          | 0.7916 | 0.8129  | 0.8257    |
| 0.0086        | 30.9994 | 38037 | 0.7777          | 0.7775 | 0.8218  | 0.8361    |
| 0.0076        | 31.9994 | 39264 | 0.8851          | 0.8851 | 0.8045  | 0.8215    |
| 0.0113        | 32.9994 | 40491 | 0.8018          | 0.8015 | 0.8137  | 0.8272    |
| 0.0124        | 33.9994 | 41718 | 0.7979          | 0.7980 | 0.8165  | 0.8307    |
| 0.0087        | 34.9994 | 42945 | 0.7377          | 0.7378 | 0.8345  | 0.8448    |
| 0.0068        | 35.9994 | 44172 | 0.8783          | 0.8779 | 0.7969  | 0.8078    |
| 0.0135        | 36.9994 | 45399 | 0.9240          | 0.9239 | 0.7956  | 0.8040    |
| 0.0094        | 37.9994 | 46626 | 0.9686          | 0.9685 | 0.7886  | 0.8031    |
| 0.0095        | 38.9994 | 47853 | 0.8474          | 0.8473 | 0.8001  | 0.8139    |
| 0.0121        | 39.9994 | 49080 | 0.8876          | 0.8874 | 0.7938  | 0.8053    |


### Framework versions

- Transformers 4.50.0.dev0
- Pytorch 2.6.0+cu118
- Datasets 3.3.2
- Tokenizers 0.21.0
