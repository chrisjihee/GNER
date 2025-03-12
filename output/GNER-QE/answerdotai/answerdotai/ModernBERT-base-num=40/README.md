---
library_name: transformers
license: apache-2.0
base_model: answerdotai/ModernBERT-base
tags:
- generated_from_trainer
metrics:
- spearmanr
model-index:
- name: ModernBERT-base-num=40
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# ModernBERT-base-num=40

This model is a fine-tuned version of [answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.7858
- Mse: 0.7856
- Pearson: 0.7990
- Spearmanr: 0.8092

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

| Training Loss | Epoch   | Step   | Validation Loss | Mse    | Pearson | Spearmanr |
|:-------------:|:-------:|:------:|:---------------:|:------:|:-------:|:---------:|
| 3.2533        | 1.0     | 3268   | 1.0962          | 1.0959 | 0.6677  | 0.6836    |
| 2.1672        | 2.0     | 6536   | 1.0810          | 1.0805 | 0.6896  | 0.7000    |
| 1.4535        | 3.0     | 9804   | 0.8683          | 0.8682 | 0.7519  | 0.7620    |
| 0.8014        | 4.0     | 13072  | 0.8247          | 0.8247 | 0.7642  | 0.7751    |
| 0.408         | 5.0     | 16340  | 0.9811          | 0.9809 | 0.7313  | 0.7430    |
| 0.4046        | 6.0     | 19608  | 1.0097          | 1.0093 | 0.7215  | 0.7359    |
| 0.3138        | 7.0     | 22876  | 0.8376          | 0.8376 | 0.7667  | 0.7793    |
| 0.2023        | 8.0     | 26144  | 0.9505          | 0.9508 | 0.7383  | 0.7509    |
| 0.1363        | 9.0     | 29412  | 0.9701          | 0.9704 | 0.7359  | 0.7507    |
| 0.1457        | 10.0    | 32680  | 0.8641          | 0.8636 | 0.7795  | 0.7855    |
| 0.1439        | 11.0    | 35948  | 0.9524          | 0.9523 | 0.7432  | 0.7648    |
| 0.1136        | 12.0    | 39216  | 0.8891          | 0.8890 | 0.7599  | 0.7680    |
| 0.0759        | 13.0    | 42484  | 0.8621          | 0.8619 | 0.7771  | 0.7866    |
| 0.0949        | 14.0    | 45752  | 0.9011          | 0.9012 | 0.7635  | 0.7724    |
| 0.1016        | 15.0    | 49020  | 0.8784          | 0.8784 | 0.7743  | 0.7825    |
| 0.0687        | 16.0    | 52288  | 0.7858          | 0.7856 | 0.7990  | 0.8092    |
| 0.0465        | 17.0    | 55556  | 0.7889          | 0.7887 | 0.7866  | 0.7948    |
| 0.0679        | 18.0    | 58824  | 0.8767          | 0.8768 | 0.7667  | 0.7733    |
| 0.0716        | 19.0    | 62092  | 0.9258          | 0.9261 | 0.7541  | 0.7717    |
| 0.0507        | 20.0    | 65360  | 0.8771          | 0.8771 | 0.7646  | 0.7793    |
| 0.0443        | 21.0    | 68628  | 0.8622          | 0.8621 | 0.7754  | 0.7868    |
| 0.0633        | 22.0    | 71896  | 0.8797          | 0.8799 | 0.7682  | 0.7770    |
| 0.0595        | 23.0    | 75164  | 0.7873          | 0.7874 | 0.7974  | 0.8044    |
| 0.0393        | 24.0    | 78432  | 0.8446          | 0.8444 | 0.7739  | 0.7863    |
| 0.0292        | 25.0    | 81700  | 0.8558          | 0.8557 | 0.7726  | 0.7794    |
| 0.0348        | 26.0    | 84968  | 1.0258          | 1.0257 | 0.7357  | 0.7494    |
| 0.0364        | 27.0    | 88236  | 1.0214          | 1.0216 | 0.7249  | 0.7325    |
| 0.0211        | 28.0    | 91504  | 0.9170          | 0.9172 | 0.7568  | 0.7674    |
| 0.0202        | 29.0    | 94772  | 1.0040          | 1.0038 | 0.7351  | 0.7517    |
| 0.0436        | 30.0    | 98040  | 0.9233          | 0.9235 | 0.7681  | 0.7821    |
| 0.0353        | 31.0    | 101308 | 0.9532          | 0.9534 | 0.7570  | 0.7706    |
| 0.0313        | 32.0    | 104576 | 0.9889          | 0.9888 | 0.7367  | 0.7503    |
| 0.0214        | 33.0    | 107844 | 0.9349          | 0.9348 | 0.7556  | 0.7674    |
| 0.0194        | 34.0    | 111112 | 0.8786          | 0.8783 | 0.7626  | 0.7739    |
| 0.0366        | 35.0    | 114380 | 0.7878          | 0.7881 | 0.7939  | 0.8033    |
| 0.0309        | 36.0    | 117648 | 0.9313          | 0.9310 | 0.7497  | 0.7532    |
| 0.0181        | 37.0    | 120916 | 0.8346          | 0.8346 | 0.7853  | 0.7957    |
| 0.0298        | 38.0    | 124184 | 0.8656          | 0.8654 | 0.7756  | 0.7900    |
| 0.0358        | 39.0    | 127452 | 0.8456          | 0.8456 | 0.7814  | 0.7925    |
| 0.0186        | 39.9878 | 130680 | 0.9674          | 0.9676 | 0.7480  | 0.7624    |


### Framework versions

- Transformers 4.50.0.dev0
- Pytorch 2.6.0+cu118
- Datasets 3.3.2
- Tokenizers 0.21.0
