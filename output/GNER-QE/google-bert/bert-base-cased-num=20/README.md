---
library_name: transformers
license: apache-2.0
base_model: google-bert/bert-base-cased
tags:
- generated_from_trainer
metrics:
- spearmanr
model-index:
- name: bert-base-cased-num=20
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# bert-base-cased-num=20

This model is a fine-tuned version of [google-bert/bert-base-cased](https://huggingface.co/google-bert/bert-base-cased) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 1.1203
- Mse: 1.1201
- Pearson: 0.7275
- Spearmanr: 0.7452

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
| 1.1688        | 0.9994  | 820   | 1.3198          | 1.3200 | 0.6371  | 0.6512    |
| 0.7354        | 1.9994  | 1640  | 1.2888          | 1.2885 | 0.6586  | 0.6557    |
| 0.6149        | 2.9994  | 2460  | 1.2490          | 1.2488 | 0.6491  | 0.6437    |
| 0.5715        | 3.9994  | 3280  | 1.4838          | 1.4835 | 0.6138  | 0.6055    |
| 0.439         | 4.9994  | 4100  | 1.2971          | 1.2971 | 0.6336  | 0.6336    |
| 0.347         | 5.9994  | 4920  | 1.2452          | 1.2450 | 0.6742  | 0.6750    |
| 0.3063        | 6.9994  | 5740  | 1.3838          | 1.3837 | 0.6593  | 0.6700    |
| 0.3088        | 7.9994  | 6560  | 1.1938          | 1.1933 | 0.6806  | 0.6914    |
| 0.2337        | 8.9994  | 7380  | 1.1952          | 1.1949 | 0.6916  | 0.7045    |
| 0.1825        | 9.9994  | 8200  | 1.0376          | 1.0372 | 0.7270  | 0.7393    |
| 0.1691        | 10.9994 | 9020  | 1.1526          | 1.1520 | 0.6960  | 0.7085    |
| 0.1806        | 11.9994 | 9840  | 1.2778          | 1.2774 | 0.6567  | 0.6802    |
| 0.1373        | 12.9994 | 10660 | 1.2193          | 1.2190 | 0.6951  | 0.7174    |
| 0.1064        | 13.9994 | 11480 | 1.0795          | 1.0793 | 0.7183  | 0.7385    |
| 0.1003        | 14.9994 | 12300 | 1.1537          | 1.1530 | 0.7173  | 0.7350    |
| 0.1096        | 15.9994 | 13120 | 1.0832          | 1.0831 | 0.7165  | 0.7405    |
| 0.0857        | 16.9994 | 13940 | 1.1528          | 1.1522 | 0.7116  | 0.7350    |
| 0.0689        | 17.9994 | 14760 | 1.1378          | 1.1375 | 0.7148  | 0.7416    |
| 0.0672        | 18.9994 | 15580 | 1.0342          | 1.0335 | 0.7376  | 0.7581    |
| 0.0754        | 19.9994 | 16400 | 1.0968          | 1.0962 | 0.7283  | 0.7451    |
| 0.0547        | 20.9994 | 17220 | 1.1498          | 1.1493 | 0.7015  | 0.7368    |
| 0.0447        | 21.9994 | 18040 | 1.1120          | 1.1110 | 0.7207  | 0.7419    |
| 0.044         | 22.9994 | 18860 | 1.1588          | 1.1582 | 0.7078  | 0.7339    |
| 0.0497        | 23.9994 | 19680 | 1.1781          | 1.1777 | 0.7325  | 0.7472    |
| 0.0396        | 24.9994 | 20500 | 1.1286          | 1.1281 | 0.7176  | 0.7442    |
| 0.0326        | 25.9994 | 21320 | 1.1399          | 1.1394 | 0.7129  | 0.7375    |
| 0.0328        | 26.9994 | 22140 | 1.0656          | 1.0654 | 0.7357  | 0.7638    |
| 0.0363        | 27.9994 | 22960 | 1.0273          | 1.0273 | 0.7437  | 0.7654    |
| 0.0296        | 28.9994 | 23780 | 1.0634          | 1.0636 | 0.7357  | 0.7564    |
| 0.0258        | 29.9994 | 24600 | 1.1256          | 1.1255 | 0.7177  | 0.7367    |
| 0.0259        | 30.9994 | 25420 | 1.0835          | 1.0830 | 0.7305  | 0.7472    |
| 0.0278        | 31.9994 | 26240 | 1.1666          | 1.1666 | 0.7218  | 0.7363    |
| 0.0257        | 32.9994 | 27060 | 1.0299          | 1.0297 | 0.7437  | 0.7669    |
| 0.0258        | 33.9994 | 27880 | 1.0051          | 1.0049 | 0.7451  | 0.7728    |
| 0.0237        | 34.9994 | 28700 | 1.1436          | 1.1436 | 0.7219  | 0.7470    |
| 0.0264        | 35.9994 | 29520 | 1.0621          | 1.0615 | 0.7363  | 0.7611    |
| 0.0223        | 36.9994 | 30340 | 1.0479          | 1.0477 | 0.7394  | 0.7644    |
| 0.0188        | 37.9994 | 31160 | 0.9162          | 0.9159 | 0.7735  | 0.7880    |
| 0.0182        | 38.9994 | 31980 | 1.0378          | 1.0374 | 0.7431  | 0.7620    |
| 0.0252        | 39.9994 | 32800 | 1.1203          | 1.1201 | 0.7275  | 0.7452    |


### Framework versions

- Transformers 4.50.0.dev0
- Pytorch 2.6.0+cu118
- Datasets 3.3.2
- Tokenizers 0.21.0
