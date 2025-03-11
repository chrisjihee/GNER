---
library_name: transformers
license: mit
base_model: microsoft/deberta-v3-large
tags:
- generated_from_trainer
metrics:
- spearmanr
model-index:
- name: deberta-v3-large-num=40
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# deberta-v3-large-num=40

This model is a fine-tuned version of [microsoft/deberta-v3-large](https://huggingface.co/microsoft/deberta-v3-large) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.5944
- Mse: 0.5945
- Pearson: 0.8570
- Spearmanr: 0.8710

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
| 0.7785        | 1.0   | 1634  | 0.8314          | 0.8314 | 0.8003  | 0.8066    |
| 0.2495        | 2.0   | 3268  | 0.7693          | 0.7694 | 0.8117  | 0.8157    |
| 0.1074        | 3.0   | 4902  | 0.6993          | 0.6995 | 0.8340  | 0.8471    |
| 0.0594        | 4.0   | 6536  | 0.6255          | 0.6253 | 0.8423  | 0.8535    |
| 0.0417        | 5.0   | 8170  | 0.6261          | 0.6259 | 0.8414  | 0.8525    |
| 0.0332        | 6.0   | 9804  | 0.6721          | 0.6720 | 0.8376  | 0.8451    |
| 0.024         | 7.0   | 11438 | 0.7452          | 0.7456 | 0.8192  | 0.8197    |
| 0.027         | 8.0   | 13072 | 0.8221          | 0.8220 | 0.8051  | 0.8143    |
| 0.0209        | 9.0   | 14706 | 0.6374          | 0.6374 | 0.8565  | 0.8658    |
| 0.0194        | 10.0  | 16340 | 0.7556          | 0.7554 | 0.8350  | 0.8420    |
| 0.0163        | 11.0  | 17974 | 0.7725          | 0.7725 | 0.8171  | 0.8312    |
| 0.0169        | 12.0  | 19608 | 0.7581          | 0.7582 | 0.8239  | 0.8347    |
| 0.0199        | 13.0  | 21242 | 0.7776          | 0.7774 | 0.8277  | 0.8363    |
| 0.0131        | 14.0  | 22876 | 0.5944          | 0.5945 | 0.8570  | 0.8710    |
| 0.018         | 15.0  | 24510 | 0.6796          | 0.6797 | 0.8373  | 0.8485    |
| 0.0113        | 16.0  | 26144 | 0.7499          | 0.7497 | 0.8278  | 0.8416    |
| 0.0133        | 17.0  | 27778 | 0.7313          | 0.7313 | 0.8286  | 0.8402    |
| 0.0107        | 18.0  | 29412 | 0.7345          | 0.7346 | 0.8252  | 0.8348    |
| 0.0113        | 19.0  | 31046 | 0.7333          | 0.7328 | 0.8312  | 0.8387    |
| 0.0104        | 20.0  | 32680 | 0.7800          | 0.7796 | 0.8111  | 0.8226    |
| 0.0091        | 21.0  | 34314 | 0.8181          | 0.8178 | 0.8098  | 0.8217    |
| 0.0109        | 22.0  | 35948 | 0.8613          | 0.8613 | 0.8027  | 0.8130    |
| 0.011         | 23.0  | 37582 | 0.6848          | 0.6848 | 0.8384  | 0.8441    |
| 0.0076        | 24.0  | 39216 | 0.7502          | 0.7500 | 0.8172  | 0.8243    |
| 0.0106        | 25.0  | 40850 | 0.7768          | 0.7767 | 0.8158  | 0.8274    |
| 0.0088        | 26.0  | 42484 | 0.8042          | 0.8042 | 0.8026  | 0.8151    |
| 0.0118        | 27.0  | 44118 | 0.7473          | 0.7472 | 0.8179  | 0.8347    |
| 0.0083        | 28.0  | 45752 | 0.8142          | 0.8141 | 0.8064  | 0.8193    |
| 0.0094        | 29.0  | 47386 | 0.7686          | 0.7683 | 0.8077  | 0.8198    |
| 0.0169        | 30.0  | 49020 | 0.9574          | 0.9573 | 0.7855  | 0.7958    |
| 0.0205        | 31.0  | 50654 | 0.7764          | 0.7764 | 0.8101  | 0.8224    |
| 0.0109        | 32.0  | 52288 | 0.6859          | 0.6857 | 0.8487  | 0.8553    |
| 0.0155        | 33.0  | 53922 | 0.7277          | 0.7275 | 0.8224  | 0.8326    |
| 0.0083        | 34.0  | 55556 | 0.6206          | 0.6204 | 0.8444  | 0.8530    |
| 0.0075        | 35.0  | 57190 | 0.6087          | 0.6086 | 0.8480  | 0.8613    |
| 0.0091        | 36.0  | 58824 | 0.5603          | 0.5601 | 0.8559  | 0.8646    |
| 0.0088        | 37.0  | 60458 | 0.5961          | 0.5959 | 0.8515  | 0.8621    |
| 0.0088        | 38.0  | 62092 | 0.7230          | 0.7228 | 0.8318  | 0.8469    |
| 0.0107        | 39.0  | 63726 | 0.7287          | 0.7284 | 0.8355  | 0.8511    |
| 0.0088        | 40.0  | 65360 | 0.6717          | 0.6718 | 0.8405  | 0.8616    |


### Framework versions

- Transformers 4.50.0.dev0
- Pytorch 2.6.0+cu118
- Datasets 3.3.2
- Tokenizers 0.21.0
