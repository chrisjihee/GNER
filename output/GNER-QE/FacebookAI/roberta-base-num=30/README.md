---
library_name: transformers
license: mit
base_model: FacebookAI/roberta-base
tags:
- generated_from_trainer
metrics:
- spearmanr
model-index:
- name: roberta-base-num=30
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# roberta-base-num=30

This model is a fine-tuned version of [FacebookAI/roberta-base](https://huggingface.co/FacebookAI/roberta-base) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.9767
- Mse: 0.9766
- Pearson: 0.7554
- Spearmanr: 0.7669

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
| 1.0434        | 0.9994  | 1227  | 1.3612          | 1.3605 | 0.6210  | 0.6309    |
| 0.6976        | 1.9994  | 2454  | 1.2988          | 1.2978 | 0.6234  | 0.6341    |
| 0.529         | 2.9994  | 3681  | 1.1191          | 1.1184 | 0.6827  | 0.6860    |
| 0.4588        | 3.9994  | 4908  | 1.2999          | 1.2995 | 0.6473  | 0.6592    |
| 0.3665        | 4.9994  | 6135  | 1.1432          | 1.1425 | 0.6774  | 0.6821    |
| 0.3243        | 5.9994  | 7362  | 1.1057          | 1.1058 | 0.6891  | 0.6905    |
| 0.2571        | 6.9994  | 8589  | 1.1960          | 1.1957 | 0.6974  | 0.7085    |
| 0.2299        | 7.9994  | 9816  | 1.0847          | 1.0843 | 0.6940  | 0.7052    |
| 0.1897        | 8.9994  | 11043 | 1.1405          | 1.1398 | 0.6853  | 0.7030    |
| 0.1713        | 9.9994  | 12270 | 1.0498          | 1.0494 | 0.7268  | 0.7376    |
| 0.1355        | 10.9994 | 13497 | 1.1087          | 1.1082 | 0.7274  | 0.7345    |
| 0.1279        | 11.9994 | 14724 | 1.1699          | 1.1696 | 0.7161  | 0.7242    |
| 0.0995        | 12.9994 | 15951 | 0.9658          | 0.9657 | 0.7384  | 0.7523    |
| 0.0971        | 13.9994 | 17178 | 1.0668          | 1.0665 | 0.7329  | 0.7393    |
| 0.0791        | 14.9994 | 18405 | 1.1178          | 1.1175 | 0.7075  | 0.7221    |
| 0.0741        | 15.9994 | 19632 | 1.0902          | 1.0902 | 0.7270  | 0.7419    |
| 0.063         | 16.9994 | 20859 | 1.0832          | 1.0829 | 0.7193  | 0.7314    |
| 0.06          | 17.9994 | 22086 | 1.0389          | 1.0384 | 0.7276  | 0.7411    |
| 0.0518        | 18.9994 | 23313 | 0.9703          | 0.9702 | 0.7534  | 0.7645    |
| 0.0482        | 19.9994 | 24540 | 1.0542          | 1.0537 | 0.7234  | 0.7426    |
| 0.0445        | 20.9994 | 25767 | 0.9442          | 0.9443 | 0.7510  | 0.7653    |
| 0.0426        | 21.9994 | 26994 | 1.1768          | 1.1763 | 0.7063  | 0.7183    |
| 0.0359        | 22.9994 | 28221 | 1.1355          | 1.1351 | 0.7118  | 0.7301    |
| 0.0354        | 23.9994 | 29448 | 0.9889          | 0.9889 | 0.7507  | 0.7637    |
| 0.0318        | 24.9994 | 30675 | 1.0122          | 1.0122 | 0.7400  | 0.7506    |
| 0.0332        | 25.9994 | 31902 | 1.0262          | 1.0260 | 0.7423  | 0.7563    |
| 0.0283        | 26.9994 | 33129 | 1.1641          | 1.1639 | 0.7009  | 0.7265    |
| 0.0268        | 27.9994 | 34356 | 1.1101          | 1.1098 | 0.7153  | 0.7309    |
| 0.0264        | 28.9994 | 35583 | 1.0160          | 1.0158 | 0.7519  | 0.7669    |
| 0.0246        | 29.9994 | 36810 | 1.1176          | 1.1179 | 0.7167  | 0.7236    |
| 0.0232        | 30.9994 | 38037 | 1.0075          | 1.0073 | 0.7362  | 0.7469    |
| 0.0243        | 31.9994 | 39264 | 1.0115          | 1.0116 | 0.7437  | 0.7555    |
| 0.0207        | 32.9994 | 40491 | 1.0581          | 1.0581 | 0.7369  | 0.7454    |
| 0.0227        | 33.9994 | 41718 | 1.0330          | 1.0329 | 0.7392  | 0.7489    |
| 0.0203        | 34.9994 | 42945 | 1.0821          | 1.0818 | 0.7276  | 0.7333    |
| 0.0191        | 35.9994 | 44172 | 1.0643          | 1.0641 | 0.7353  | 0.7443    |
| 0.02          | 36.9994 | 45399 | 0.9971          | 0.9968 | 0.7540  | 0.7629    |
| 0.0203        | 37.9994 | 46626 | 1.0270          | 1.0270 | 0.7414  | 0.7446    |
| 0.0171        | 38.9994 | 47853 | 1.0737          | 1.0737 | 0.7325  | 0.7465    |
| 0.0165        | 39.9994 | 49080 | 0.9767          | 0.9766 | 0.7554  | 0.7669    |


### Framework versions

- Transformers 4.50.0.dev0
- Pytorch 2.6.0+cu118
- Datasets 3.3.2
- Tokenizers 0.21.0
