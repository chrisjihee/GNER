---
library_name: transformers
license: apache-2.0
base_model: answerdotai/ModernBERT-base
tags:
- generated_from_trainer
metrics:
- spearmanr
model-index:
- name: ModernBERT-base-num=10
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# ModernBERT-base-num=10

This model is a fine-tuned version of [answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 1.0087
- Mse: 1.0086
- Pearson: 0.7421
- Spearmanr: 0.7513

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
- gradient_accumulation_steps: 4
- total_train_batch_size: 16
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- num_epochs: 40.0

### Training results

| Training Loss | Epoch   | Step  | Validation Loss | Mse    | Pearson | Spearmanr |
|:-------------:|:-------:|:-----:|:---------------:|:------:|:-------:|:---------:|
| 5.4585        | 0.9991  | 823   | 1.3824          | 1.3822 | 0.6096  | 0.6267    |
| 3.2247        | 1.9991  | 1646  | 1.1916          | 1.1918 | 0.6767  | 0.6852    |
| 2.721         | 2.9991  | 2469  | 1.2985          | 1.2986 | 0.6540  | 0.6683    |
| 2.5899        | 3.9991  | 3292  | 1.3198          | 1.3199 | 0.6345  | 0.6388    |
| 1.9547        | 4.9991  | 4115  | 1.3627          | 1.3620 | 0.6453  | 0.6534    |
| 1.4046        | 5.9991  | 4938  | 1.4113          | 1.4119 | 0.6517  | 0.6484    |
| 1.2439        | 6.9991  | 5761  | 1.2613          | 1.2605 | 0.6539  | 0.6605    |
| 1.2642        | 7.9991  | 6584  | 1.0863          | 1.0857 | 0.7147  | 0.7143    |
| 0.9595        | 8.9991  | 7407  | 1.1054          | 1.1052 | 0.7077  | 0.7136    |
| 0.6898        | 9.9991  | 8230  | 1.1546          | 1.1536 | 0.6906  | 0.6913    |
| 0.612         | 10.9991 | 9053  | 1.1017          | 1.1014 | 0.7115  | 0.7195    |
| 0.7014        | 11.9991 | 9876  | 1.1798          | 1.1801 | 0.6883  | 0.6928    |
| 0.5023        | 12.9991 | 10699 | 1.2111          | 1.2109 | 0.6829  | 0.6950    |
| 0.3629        | 13.9991 | 11522 | 1.1693          | 1.1697 | 0.6908  | 0.7012    |
| 0.3627        | 14.9991 | 12345 | 1.0617          | 1.0617 | 0.7207  | 0.7270    |
| 0.4408        | 15.9991 | 13168 | 1.1459          | 1.1453 | 0.7024  | 0.7052    |
| 0.3081        | 16.9991 | 13991 | 1.0539          | 1.0543 | 0.7330  | 0.7434    |
| 0.2423        | 17.9991 | 14814 | 1.1827          | 1.1822 | 0.7104  | 0.7176    |
| 0.2297        | 18.9991 | 15637 | 1.0850          | 1.0852 | 0.7176  | 0.7258    |
| 0.2756        | 19.9991 | 16460 | 1.1824          | 1.1821 | 0.7130  | 0.7217    |
| 0.2199        | 20.9991 | 17283 | 1.0087          | 1.0086 | 0.7421  | 0.7513    |
| 0.1521        | 21.9991 | 18106 | 1.2715          | 1.2710 | 0.6698  | 0.6753    |
| 0.1553        | 22.9991 | 18929 | 1.2177          | 1.2171 | 0.6915  | 0.7019    |
| 0.2099        | 23.9991 | 19752 | 1.3299          | 1.3306 | 0.7143  | 0.7275    |
| 0.1521        | 24.9991 | 20575 | 1.1861          | 1.1853 | 0.6954  | 0.7001    |
| 0.1194        | 25.9991 | 21398 | 1.1492          | 1.1483 | 0.7038  | 0.7083    |
| 0.1192        | 26.9991 | 22221 | 1.3027          | 1.3024 | 0.6718  | 0.6882    |
| 0.1375        | 27.9991 | 23044 | 1.3609          | 1.3609 | 0.6709  | 0.6817    |
| 0.1049        | 28.9991 | 23867 | 1.1937          | 1.1954 | 0.6940  | 0.6992    |
| 0.0943        | 29.9991 | 24690 | 1.1523          | 1.1520 | 0.7071  | 0.7126    |
| 0.1016        | 30.9991 | 25513 | 1.3360          | 1.3360 | 0.6628  | 0.6641    |
| 0.1042        | 31.9991 | 26336 | 1.4192          | 1.4195 | 0.6413  | 0.6541    |
| 0.0856        | 32.9991 | 27159 | 1.2150          | 1.2152 | 0.6935  | 0.7019    |
| 0.0752        | 33.9991 | 27982 | 1.4252          | 1.4255 | 0.6375  | 0.6492    |
| 0.0787        | 34.9991 | 28805 | 1.2816          | 1.2814 | 0.6724  | 0.6849    |
| 0.0831        | 35.9991 | 29628 | 1.3153          | 1.3146 | 0.6672  | 0.6723    |
| 0.0871        | 36.9991 | 30451 | 1.1786          | 1.1787 | 0.7121  | 0.7242    |
| 0.0519        | 37.9991 | 31274 | 1.2226          | 1.2224 | 0.7031  | 0.7109    |
| 0.0487        | 38.9991 | 32097 | 1.1022          | 1.1022 | 0.7232  | 0.7364    |
| 0.0778        | 39.9991 | 32920 | 1.2786          | 1.2796 | 0.6803  | 0.6919    |


### Framework versions

- Transformers 4.50.0.dev0
- Pytorch 2.6.0+cu118
- Datasets 3.3.2
- Tokenizers 0.21.0
