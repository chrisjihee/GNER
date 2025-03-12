---
library_name: transformers
license: apache-2.0
base_model: answerdotai/ModernBERT-base
tags:
- generated_from_trainer
metrics:
- spearmanr
model-index:
- name: ModernBERT-base-num=20
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# ModernBERT-base-num=20

This model is a fine-tuned version of [answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.8762
- Mse: 0.8765
- Pearson: 0.7831
- Spearmanr: 0.7947

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

| Training Loss | Epoch | Step  | Validation Loss | Mse    | Pearson | Spearmanr |
|:-------------:|:-----:|:-----:|:---------------:|:------:|:-------:|:---------:|
| 4.2448        | 1.0   | 1641  | 1.2469          | 1.2471 | 0.6641  | 0.6743    |
| 2.2742        | 2.0   | 3282  | 1.1232          | 1.1232 | 0.6862  | 0.6954    |
| 1.5294        | 3.0   | 4923  | 1.0577          | 1.0577 | 0.7087  | 0.7166    |
| 1.0284        | 4.0   | 6564  | 1.0833          | 1.0830 | 0.7038  | 0.7165    |
| 0.6947        | 5.0   | 8205  | 1.2185          | 1.2184 | 0.6988  | 0.7089    |
| 0.484         | 6.0   | 9846  | 0.9833          | 0.9837 | 0.7459  | 0.7588    |
| 0.3391        | 7.0   | 11487 | 1.1577          | 1.1581 | 0.7108  | 0.7232    |
| 0.2634        | 8.0   | 13128 | 1.0768          | 1.0768 | 0.7225  | 0.7333    |
| 0.1889        | 9.0   | 14769 | 1.0212          | 1.0214 | 0.7307  | 0.7413    |
| 0.1532        | 10.0  | 16410 | 0.9922          | 0.9922 | 0.7441  | 0.7572    |
| 0.1253        | 11.0  | 18051 | 0.9084          | 0.9085 | 0.7680  | 0.7790    |
| 0.0995        | 12.0  | 19692 | 0.9594          | 0.9594 | 0.7514  | 0.7649    |
| 0.0835        | 13.0  | 21333 | 1.0218          | 1.0216 | 0.7409  | 0.7542    |
| 0.0743        | 14.0  | 22974 | 0.9893          | 0.9893 | 0.7408  | 0.7557    |
| 0.0651        | 15.0  | 24615 | 1.0696          | 1.0702 | 0.7154  | 0.7312    |
| 0.06          | 16.0  | 26256 | 0.9838          | 0.9842 | 0.7445  | 0.7539    |
| 0.0537        | 17.0  | 27897 | 0.9557          | 0.9557 | 0.7622  | 0.7708    |
| 0.0467        | 18.0  | 29538 | 1.0010          | 1.0011 | 0.7409  | 0.7536    |
| 0.0376        | 19.0  | 31179 | 0.9584          | 0.9587 | 0.7533  | 0.7662    |
| 0.0362        | 20.0  | 32820 | 0.9527          | 0.9528 | 0.7533  | 0.7673    |
| 0.0372        | 21.0  | 34461 | 1.1100          | 1.1099 | 0.7203  | 0.7352    |
| 0.0359        | 22.0  | 36102 | 1.0835          | 1.0837 | 0.7169  | 0.7270    |
| 0.0356        | 23.0  | 37743 | 1.0329          | 1.0336 | 0.7380  | 0.7514    |
| 0.0357        | 24.0  | 39384 | 1.0766          | 1.0769 | 0.7251  | 0.7445    |
| 0.0327        | 25.0  | 41025 | 0.9927          | 0.9929 | 0.7449  | 0.7563    |
| 0.0315        | 26.0  | 42666 | 1.0184          | 1.0188 | 0.7396  | 0.7554    |
| 0.0268        | 27.0  | 44307 | 1.0482          | 1.0485 | 0.7385  | 0.7563    |
| 0.0289        | 28.0  | 45948 | 1.3452          | 1.3456 | 0.6591  | 0.6995    |
| 0.0266        | 29.0  | 47589 | 0.9849          | 0.9847 | 0.7485  | 0.7625    |
| 0.0257        | 30.0  | 49230 | 1.1061          | 1.1060 | 0.7193  | 0.7350    |
| 0.0211        | 31.0  | 50871 | 1.0024          | 1.0026 | 0.7441  | 0.7542    |
| 0.0228        | 32.0  | 52512 | 0.9750          | 0.9747 | 0.7561  | 0.7658    |
| 0.022         | 33.0  | 54153 | 1.0076          | 1.0078 | 0.7421  | 0.7501    |
| 0.0218        | 34.0  | 55794 | 1.0208          | 1.0210 | 0.7424  | 0.7622    |
| 0.0238        | 35.0  | 57435 | 1.0513          | 1.0519 | 0.7335  | 0.7480    |
| 0.0243        | 36.0  | 59076 | 0.9833          | 0.9834 | 0.7515  | 0.7666    |
| 0.0182        | 37.0  | 60717 | 1.0078          | 1.0079 | 0.7460  | 0.7635    |
| 0.0194        | 38.0  | 62358 | 0.8762          | 0.8765 | 0.7831  | 0.7947    |
| 0.0156        | 39.0  | 63999 | 0.9908          | 0.9907 | 0.7497  | 0.7692    |
| 0.0276        | 40.0  | 65640 | 1.0256          | 1.0258 | 0.7408  | 0.7597    |


### Framework versions

- Transformers 4.50.0.dev0
- Pytorch 2.6.0+cu118
- Datasets 3.3.2
- Tokenizers 0.21.0
