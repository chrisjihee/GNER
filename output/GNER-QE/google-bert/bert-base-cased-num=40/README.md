---
library_name: transformers
license: apache-2.0
base_model: google-bert/bert-base-cased
tags:
- generated_from_trainer
metrics:
- spearmanr
model-index:
- name: bert-base-cased-num=40
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# bert-base-cased-num=40

This model is a fine-tuned version of [google-bert/bert-base-cased](https://huggingface.co/google-bert/bert-base-cased) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.9223
- Mse: 0.9223
- Pearson: 0.7670
- Spearmanr: 0.7868

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
| 0.9156        | 1.0   | 1634  | 1.0694          | 1.0692 | 0.6858  | 0.6925    |
| 0.4858        | 2.0   | 3268  | 1.1467          | 1.1471 | 0.6950  | 0.7010    |
| 0.3098        | 3.0   | 4902  | 0.9209          | 0.9210 | 0.7428  | 0.7510    |
| 0.2085        | 4.0   | 6536  | 1.0072          | 1.0070 | 0.7187  | 0.7275    |
| 0.1389        | 5.0   | 8170  | 0.9026          | 0.9024 | 0.7515  | 0.7644    |
| 0.0988        | 6.0   | 9804  | 0.9994          | 0.9996 | 0.7507  | 0.7634    |
| 0.0721        | 7.0   | 11438 | 0.9644          | 0.9642 | 0.7419  | 0.7546    |
| 0.0554        | 8.0   | 13072 | 0.9871          | 0.9868 | 0.7281  | 0.7467    |
| 0.0443        | 9.0   | 14706 | 0.9548          | 0.9548 | 0.7413  | 0.7543    |
| 0.035         | 10.0  | 16340 | 0.9187          | 0.9187 | 0.7607  | 0.7675    |
| 0.0336        | 11.0  | 17974 | 0.9171          | 0.9165 | 0.7598  | 0.7715    |
| 0.0297        | 12.0  | 19608 | 0.8850          | 0.8848 | 0.7632  | 0.7824    |
| 0.0257        | 13.0  | 21242 | 0.9342          | 0.9341 | 0.7575  | 0.7679    |
| 0.0227        | 14.0  | 22876 | 1.0096          | 1.0094 | 0.7534  | 0.7608    |
| 0.0227        | 15.0  | 24510 | 0.9548          | 0.9550 | 0.7484  | 0.7692    |
| 0.0209        | 16.0  | 26144 | 0.9005          | 0.9007 | 0.7652  | 0.7792    |
| 0.0205        | 17.0  | 27778 | 0.9463          | 0.9464 | 0.7475  | 0.7562    |
| 0.0177        | 18.0  | 29412 | 0.8329          | 0.8326 | 0.7797  | 0.7942    |
| 0.015         | 19.0  | 31046 | 0.9621          | 0.9618 | 0.7455  | 0.7565    |
| 0.0144        | 20.0  | 32680 | 0.9054          | 0.9054 | 0.7574  | 0.7698    |
| 0.0144        | 21.0  | 34314 | 0.8438          | 0.8438 | 0.7792  | 0.7873    |
| 0.0142        | 22.0  | 35948 | 0.8549          | 0.8549 | 0.7743  | 0.7845    |
| 0.0138        | 23.0  | 37582 | 0.9853          | 0.9850 | 0.7496  | 0.7611    |
| 0.0139        | 24.0  | 39216 | 0.9319          | 0.9322 | 0.7575  | 0.7632    |
| 0.0123        | 25.0  | 40850 | 1.0224          | 1.0224 | 0.7402  | 0.7548    |
| 0.0122        | 26.0  | 42484 | 0.9878          | 0.9877 | 0.7447  | 0.7576    |
| 0.0132        | 27.0  | 44118 | 0.9905          | 0.9905 | 0.7445  | 0.7574    |
| 0.0105        | 28.0  | 45752 | 0.8804          | 0.8802 | 0.7708  | 0.7837    |
| 0.012         | 29.0  | 47386 | 0.8978          | 0.8982 | 0.7676  | 0.7827    |
| 0.0113        | 30.0  | 49020 | 0.9881          | 0.9878 | 0.7401  | 0.7537    |
| 0.0105        | 31.0  | 50654 | 0.8826          | 0.8825 | 0.7677  | 0.7796    |
| 0.01          | 32.0  | 52288 | 0.9016          | 0.9014 | 0.7593  | 0.7708    |
| 0.01          | 33.0  | 53922 | 0.8521          | 0.8521 | 0.7765  | 0.7847    |
| 0.0099        | 34.0  | 55556 | 0.9338          | 0.9336 | 0.7516  | 0.7682    |
| 0.0096        | 35.0  | 57190 | 0.8776          | 0.8776 | 0.7734  | 0.7857    |
| 0.0093        | 36.0  | 58824 | 0.8661          | 0.8661 | 0.7779  | 0.7953    |
| 0.0096        | 37.0  | 60458 | 1.1034          | 1.1037 | 0.7071  | 0.7332    |
| 0.01          | 38.0  | 62092 | 0.8413          | 0.8412 | 0.7883  | 0.8009    |
| 0.0089        | 39.0  | 63726 | 0.8689          | 0.8688 | 0.7752  | 0.7837    |
| 0.0082        | 40.0  | 65360 | 0.9223          | 0.9223 | 0.7670  | 0.7868    |


### Framework versions

- Transformers 4.50.0.dev0
- Pytorch 2.6.0+cu118
- Datasets 3.3.2
- Tokenizers 0.21.0
