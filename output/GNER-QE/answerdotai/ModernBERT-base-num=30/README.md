---
library_name: transformers
license: apache-2.0
base_model: answerdotai/ModernBERT-base
tags:
- generated_from_trainer
metrics:
- spearmanr
model-index:
- name: ModernBERT-base-num=30
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# ModernBERT-base-num=30

This model is a fine-tuned version of [answerdotai/ModernBERT-base](https://huggingface.co/answerdotai/ModernBERT-base) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.8824
- Mse: 0.8823
- Pearson: 0.7701
- Spearmanr: 0.7867

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
| 3.8444        | 1.0     | 2456  | 1.2251          | 1.2251 | 0.6388  | 0.6548    |
| 2.4403        | 2.0     | 4912  | 1.5495          | 1.5496 | 0.5852  | 0.6053    |
| 1.565         | 3.0     | 7368  | 1.1649          | 1.1649 | 0.6651  | 0.6803    |
| 1.1117        | 4.0     | 9824  | 1.1980          | 1.1980 | 0.6800  | 0.6835    |
| 0.6779        | 5.0     | 12280 | 1.1067          | 1.1065 | 0.6931  | 0.6996    |
| 0.5032        | 6.0     | 14736 | 0.9965          | 0.9965 | 0.7170  | 0.7208    |
| 0.3363        | 7.0     | 17192 | 1.0463          | 1.0462 | 0.7147  | 0.7374    |
| 0.2719        | 8.0     | 19648 | 1.0574          | 1.0570 | 0.7124  | 0.7347    |
| 0.1937        | 9.0     | 22104 | 1.0128          | 1.0129 | 0.7227  | 0.7395    |
| 0.1751        | 10.0    | 24560 | 1.1368          | 1.1373 | 0.6872  | 0.7035    |
| 0.1335        | 11.0    | 27016 | 1.0500          | 1.0500 | 0.7180  | 0.7331    |
| 0.1171        | 12.0    | 29472 | 1.1995          | 1.1996 | 0.6819  | 0.7043    |
| 0.1017        | 13.0    | 31928 | 1.0472          | 1.0470 | 0.7191  | 0.7376    |
| 0.0944        | 14.0    | 34384 | 1.0764          | 1.0768 | 0.7209  | 0.7403    |
| 0.0736        | 15.0    | 36840 | 1.0913          | 1.0916 | 0.7117  | 0.7350    |
| 0.0686        | 16.0    | 39296 | 0.9023          | 0.9021 | 0.7631  | 0.7759    |
| 0.0572        | 17.0    | 41752 | 1.2073          | 1.2076 | 0.6785  | 0.6938    |
| 0.0582        | 18.0    | 44208 | 0.9251          | 0.9252 | 0.7643  | 0.7787    |
| 0.0532        | 19.0    | 46664 | 1.0578          | 1.0577 | 0.7220  | 0.7359    |
| 0.0518        | 20.0    | 49120 | 0.9034          | 0.9033 | 0.7630  | 0.7711    |
| 0.0528        | 21.0    | 51576 | 1.2343          | 1.2349 | 0.6733  | 0.6983    |
| 0.0476        | 22.0    | 54032 | 0.9682          | 0.9678 | 0.7464  | 0.7544    |
| 0.0431        | 23.0    | 56488 | 0.9855          | 0.9850 | 0.7429  | 0.7592    |
| 0.0391        | 24.0    | 58944 | 1.2849          | 1.2851 | 0.6583  | 0.6935    |
| 0.0403        | 25.0    | 61400 | 1.1548          | 1.1549 | 0.6961  | 0.7271    |
| 0.0421        | 26.0    | 63856 | 1.2330          | 1.2326 | 0.6753  | 0.7050    |
| 0.0327        | 27.0    | 66312 | 1.1028          | 1.1024 | 0.7157  | 0.7447    |
| 0.0306        | 28.0    | 68768 | 1.1009          | 1.1007 | 0.7076  | 0.7312    |
| 0.0387        | 29.0    | 71224 | 1.1294          | 1.1297 | 0.7111  | 0.7380    |
| 0.0364        | 30.0    | 73680 | 0.9879          | 0.9879 | 0.7428  | 0.7622    |
| 0.0297        | 31.0    | 76136 | 1.1749          | 1.1750 | 0.7019  | 0.7306    |
| 0.0345        | 32.0    | 78592 | 1.0194          | 1.0194 | 0.7420  | 0.7620    |
| 0.0345        | 33.0    | 81048 | 1.0276          | 1.0277 | 0.7352  | 0.7522    |
| 0.0282        | 34.0    | 83504 | 1.3276          | 1.3276 | 0.6587  | 0.6808    |
| 0.032         | 35.0    | 85960 | 1.1287          | 1.1282 | 0.7026  | 0.7216    |
| 0.0232        | 36.0    | 88416 | 0.9719          | 0.9715 | 0.7501  | 0.7704    |
| 0.0283        | 37.0    | 90872 | 0.8824          | 0.8823 | 0.7701  | 0.7867    |
| 0.024         | 38.0    | 93328 | 1.1220          | 1.1220 | 0.7192  | 0.7385    |
| 0.0219        | 39.0    | 95784 | 1.0879          | 1.0881 | 0.7202  | 0.7331    |
| 0.022         | 39.9839 | 98200 | 1.1248          | 1.1245 | 0.7096  | 0.7228    |


### Framework versions

- Transformers 4.50.0.dev0
- Pytorch 2.6.0+cu118
- Datasets 3.3.2
- Tokenizers 0.21.0
