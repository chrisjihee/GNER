---
library_name: transformers
license: mit
base_model: FacebookAI/roberta-large
tags:
- generated_from_trainer
metrics:
- spearmanr
model-index:
- name: roberta-large-num=10
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# roberta-large-num=10

This model is a fine-tuned version of [FacebookAI/roberta-large](https://huggingface.co/FacebookAI/roberta-large) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 1.3650
- Mse: 1.3648
- Pearson: 0.7193
- Spearmanr: 0.7376

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
| 1.4874        | 1.0   | 412   | 1.6188          | 1.6201 | 0.6405  | 0.6486    |
| 0.9012        | 2.0   | 824   | 1.2007          | 1.2001 | 0.6896  | 0.6958    |
| 0.7001        | 3.0   | 1236  | 1.1542          | 1.1528 | 0.7128  | 0.7214    |
| 0.6086        | 4.0   | 1648  | 1.1659          | 1.1654 | 0.7148  | 0.7238    |
| 0.5223        | 5.0   | 2060  | 1.1605          | 1.1586 | 0.6997  | 0.6950    |
| 0.445         | 6.0   | 2472  | 1.1865          | 1.1850 | 0.6976  | 0.6942    |
| 0.3884        | 7.0   | 2884  | 1.3717          | 1.3708 | 0.7104  | 0.7108    |
| 0.3264        | 8.0   | 3296  | 1.1920          | 1.1917 | 0.7108  | 0.7260    |
| 0.3024        | 9.0   | 3708  | 1.2328          | 1.2325 | 0.7138  | 0.7214    |
| 0.2564        | 10.0  | 4120  | 1.1160          | 1.1153 | 0.7216  | 0.7299    |
| 0.2266        | 11.0  | 4532  | 1.4336          | 1.4335 | 0.7109  | 0.7257    |
| 0.1988        | 12.0  | 4944  | 1.3282          | 1.3260 | 0.6889  | 0.7016    |
| 0.1806        | 13.0  | 5356  | 1.1076          | 1.1077 | 0.7430  | 0.7492    |
| 0.1593        | 14.0  | 5768  | 1.4464          | 1.4456 | 0.7150  | 0.7324    |
| 0.1416        | 15.0  | 6180  | 1.2442          | 1.2435 | 0.7084  | 0.7207    |
| 0.121         | 16.0  | 6592  | 1.2552          | 1.2550 | 0.7127  | 0.7300    |
| 0.1101        | 17.0  | 7004  | 1.2797          | 1.2797 | 0.7150  | 0.7257    |
| 0.0985        | 18.0  | 7416  | 1.2891          | 1.2886 | 0.7259  | 0.7463    |
| 0.0844        | 19.0  | 7828  | 1.3944          | 1.3938 | 0.6981  | 0.7172    |
| 0.0865        | 20.0  | 8240  | 1.4003          | 1.4006 | 0.7067  | 0.7230    |
| 0.0733        | 21.0  | 8652  | 1.4553          | 1.4553 | 0.7008  | 0.7167    |
| 0.0676        | 22.0  | 9064  | 1.2715          | 1.2723 | 0.7258  | 0.7432    |
| 0.0591        | 23.0  | 9476  | 1.2415          | 1.2417 | 0.7343  | 0.7506    |
| 0.0602        | 24.0  | 9888  | 1.5636          | 1.5626 | 0.6965  | 0.7163    |
| 0.0589        | 25.0  | 10300 | 1.4590          | 1.4593 | 0.7206  | 0.7386    |
| 0.059         | 26.0  | 10712 | 1.4579          | 1.4574 | 0.7147  | 0.7305    |
| 0.0501        | 27.0  | 11124 | 1.4941          | 1.4934 | 0.7035  | 0.7185    |
| 0.0453        | 28.0  | 11536 | 1.2864          | 1.2856 | 0.7075  | 0.7249    |
| 0.0444        | 29.0  | 11948 | 1.5277          | 1.5279 | 0.7002  | 0.7205    |
| 0.0425        | 30.0  | 12360 | 1.3561          | 1.3560 | 0.7162  | 0.7326    |
| 0.0404        | 31.0  | 12772 | 1.4498          | 1.4512 | 0.6880  | 0.7089    |
| 0.0405        | 32.0  | 13184 | 1.3546          | 1.3545 | 0.7024  | 0.7244    |
| 0.0377        | 33.0  | 13596 | 1.3892          | 1.3889 | 0.7144  | 0.7365    |
| 0.0368        | 34.0  | 14008 | 1.3607          | 1.3615 | 0.7230  | 0.7412    |
| 0.0341        | 35.0  | 14420 | 1.4790          | 1.4789 | 0.7181  | 0.7349    |
| 0.033         | 36.0  | 14832 | 1.1715          | 1.1726 | 0.7400  | 0.7599    |
| 0.033         | 37.0  | 15244 | 1.1935          | 1.1938 | 0.7508  | 0.7711    |
| 0.0302        | 38.0  | 15656 | 1.3063          | 1.3062 | 0.7386  | 0.7569    |
| 0.0271        | 39.0  | 16068 | 1.4163          | 1.4167 | 0.7249  | 0.7428    |
| 0.0295        | 40.0  | 16480 | 1.3650          | 1.3648 | 0.7193  | 0.7376    |


### Framework versions

- Transformers 4.50.0.dev0
- Pytorch 2.6.0+cu118
- Datasets 3.3.2
- Tokenizers 0.21.0
