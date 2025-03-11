---
library_name: transformers
license: mit
base_model: microsoft/deberta-v3-base
tags:
- generated_from_trainer
metrics:
- spearmanr
model-index:
- name: deberta-v3-base-num=10
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# deberta-v3-base-num=10

This model is a fine-tuned version of [microsoft/deberta-v3-base](https://huggingface.co/microsoft/deberta-v3-base) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 1.1579
- Mse: 1.1580
- Pearson: 0.7489
- Spearmanr: 0.7723

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
| 1.6949        | 1.0   | 412   | 1.5112          | 1.5132 | 0.5656  | 0.5885    |
| 0.9513        | 2.0   | 824   | 1.3679          | 1.3683 | 0.6229  | 0.6288    |
| 0.7446        | 3.0   | 1236  | 1.5138          | 1.5146 | 0.6362  | 0.6587    |
| 0.6259        | 4.0   | 1648  | 1.2792          | 1.2803 | 0.6686  | 0.6734    |
| 0.524         | 5.0   | 2060  | 1.3217          | 1.3227 | 0.6548  | 0.6625    |
| 0.4549        | 6.0   | 2472  | 1.3232          | 1.3256 | 0.6492  | 0.6727    |
| 0.3969        | 7.0   | 2884  | 1.5304          | 1.5291 | 0.6811  | 0.6887    |
| 0.3437        | 8.0   | 3296  | 1.4102          | 1.4109 | 0.6894  | 0.7049    |
| 0.3045        | 9.0   | 3708  | 1.2687          | 1.2691 | 0.7123  | 0.7179    |
| 0.2652        | 10.0  | 4120  | 1.2116          | 1.2114 | 0.6859  | 0.6999    |
| 0.2336        | 11.0  | 4532  | 1.6579          | 1.6575 | 0.6650  | 0.6756    |
| 0.2095        | 12.0  | 4944  | 1.4916          | 1.4926 | 0.6635  | 0.6755    |
| 0.1793        | 13.0  | 5356  | 1.5230          | 1.5237 | 0.6822  | 0.6990    |
| 0.1639        | 14.0  | 5768  | 1.5796          | 1.5799 | 0.6955  | 0.7128    |
| 0.1441        | 15.0  | 6180  | 1.2774          | 1.2776 | 0.7062  | 0.7261    |
| 0.1306        | 16.0  | 6592  | 1.5869          | 1.5873 | 0.6889  | 0.7015    |
| 0.1207        | 17.0  | 7004  | 1.4146          | 1.4148 | 0.6773  | 0.6940    |
| 0.1079        | 18.0  | 7416  | 1.5431          | 1.5415 | 0.6860  | 0.6991    |
| 0.0958        | 19.0  | 7828  | 1.4356          | 1.4350 | 0.6649  | 0.6871    |
| 0.091         | 20.0  | 8240  | 1.3889          | 1.3891 | 0.6881  | 0.7065    |
| 0.0819        | 21.0  | 8652  | 1.3985          | 1.3987 | 0.6715  | 0.6924    |
| 0.0777        | 22.0  | 9064  | 1.3963          | 1.3970 | 0.6834  | 0.7134    |
| 0.0697        | 23.0  | 9476  | 1.2831          | 1.2827 | 0.7031  | 0.7277    |
| 0.0666        | 24.0  | 9888  | 1.3326          | 1.3323 | 0.7161  | 0.7355    |
| 0.0579        | 25.0  | 10300 | 1.3600          | 1.3612 | 0.7118  | 0.7296    |
| 0.0529        | 26.0  | 10712 | 1.4351          | 1.4351 | 0.6986  | 0.7197    |
| 0.0481        | 27.0  | 11124 | 1.3798          | 1.3803 | 0.7087  | 0.7349    |
| 0.0467        | 28.0  | 11536 | 1.2298          | 1.2297 | 0.7438  | 0.7626    |
| 0.0416        | 29.0  | 11948 | 1.4476          | 1.4477 | 0.6866  | 0.7093    |
| 0.0378        | 30.0  | 12360 | 1.3007          | 1.3003 | 0.7158  | 0.7367    |
| 0.0352        | 31.0  | 12772 | 1.2224          | 1.2236 | 0.7223  | 0.7463    |
| 0.0349        | 32.0  | 13184 | 1.2309          | 1.2314 | 0.7295  | 0.7448    |
| 0.0313        | 33.0  | 13596 | 1.2733          | 1.2739 | 0.7301  | 0.7533    |
| 0.0295        | 34.0  | 14008 | 1.2837          | 1.2851 | 0.7242  | 0.7491    |
| 0.0303        | 35.0  | 14420 | 1.1579          | 1.1580 | 0.7489  | 0.7723    |
| 0.0262        | 36.0  | 14832 | 1.2333          | 1.2339 | 0.7333  | 0.7569    |
| 0.0293        | 37.0  | 15244 | 1.1919          | 1.1920 | 0.7382  | 0.7580    |
| 0.025         | 38.0  | 15656 | 1.1167          | 1.1162 | 0.7446  | 0.7591    |
| 0.024         | 39.0  | 16068 | 1.2306          | 1.2310 | 0.7457  | 0.7648    |
| 0.0219        | 40.0  | 16480 | 1.1408          | 1.1405 | 0.7395  | 0.7643    |


### Framework versions

- Transformers 4.50.0.dev0
- Pytorch 2.6.0+cu118
- Datasets 3.3.2
- Tokenizers 0.21.0
