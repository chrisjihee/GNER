---
library_name: transformers
license: mit
base_model: FacebookAI/roberta-base
tags:
- generated_from_trainer
metrics:
- spearmanr
model-index:
- name: roberta-base-num=10
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# roberta-base-num=10

This model is a fine-tuned version of [FacebookAI/roberta-base](https://huggingface.co/FacebookAI/roberta-base) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 1.2525
- Mse: 1.2535
- Pearson: 0.7030
- Spearmanr: 0.7149

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
| 1.6132        | 1.0   | 412   | 1.3423          | 1.3419 | 0.6239  | 0.6402    |
| 0.9431        | 2.0   | 824   | 1.4091          | 1.4092 | 0.6276  | 0.6293    |
| 0.762         | 3.0   | 1236  | 1.5082          | 1.5077 | 0.6253  | 0.6433    |
| 0.6394        | 4.0   | 1648  | 1.5449          | 1.5447 | 0.6184  | 0.6211    |
| 0.5614        | 5.0   | 2060  | 1.4091          | 1.4068 | 0.6340  | 0.6323    |
| 0.4977        | 6.0   | 2472  | 1.3568          | 1.3576 | 0.6381  | 0.6405    |
| 0.442         | 7.0   | 2884  | 1.3436          | 1.3431 | 0.6396  | 0.6471    |
| 0.3848        | 8.0   | 3296  | 1.5579          | 1.5578 | 0.6122  | 0.6231    |
| 0.3467        | 9.0   | 3708  | 1.5896          | 1.5890 | 0.6360  | 0.6330    |
| 0.3145        | 10.0  | 4120  | 1.5566          | 1.5570 | 0.6102  | 0.6241    |
| 0.2805        | 11.0  | 4532  | 1.5363          | 1.5364 | 0.6426  | 0.6525    |
| 0.2581        | 12.0  | 4944  | 1.4092          | 1.4067 | 0.6483  | 0.6601    |
| 0.2268        | 13.0  | 5356  | 1.5585          | 1.5585 | 0.6591  | 0.6626    |
| 0.2102        | 14.0  | 5768  | 1.4312          | 1.4308 | 0.6748  | 0.6882    |
| 0.1914        | 15.0  | 6180  | 1.4482          | 1.4480 | 0.6487  | 0.6586    |
| 0.1824        | 16.0  | 6592  | 1.4621          | 1.4628 | 0.6560  | 0.6713    |
| 0.1628        | 17.0  | 7004  | 1.3253          | 1.3253 | 0.6744  | 0.6822    |
| 0.1581        | 18.0  | 7416  | 1.4340          | 1.4338 | 0.6767  | 0.6886    |
| 0.1427        | 19.0  | 7828  | 1.3397          | 1.3400 | 0.6661  | 0.6787    |
| 0.1323        | 20.0  | 8240  | 1.3205          | 1.3208 | 0.6845  | 0.6993    |
| 0.1241        | 21.0  | 8652  | 1.3566          | 1.3573 | 0.6737  | 0.6869    |
| 0.113         | 22.0  | 9064  | 1.2371          | 1.2374 | 0.6867  | 0.6989    |
| 0.1105        | 23.0  | 9476  | 1.3253          | 1.3259 | 0.6589  | 0.6769    |
| 0.1           | 24.0  | 9888  | 1.4163          | 1.4171 | 0.6693  | 0.6823    |
| 0.0952        | 25.0  | 10300 | 1.3944          | 1.3945 | 0.6637  | 0.6800    |
| 0.0866        | 26.0  | 10712 | 1.3691          | 1.3699 | 0.6790  | 0.6864    |
| 0.0811        | 27.0  | 11124 | 1.3177          | 1.3178 | 0.6781  | 0.6868    |
| 0.0776        | 28.0  | 11536 | 1.2621          | 1.2628 | 0.6870  | 0.6983    |
| 0.0711        | 29.0  | 11948 | 1.2970          | 1.2966 | 0.6854  | 0.6985    |
| 0.0676        | 30.0  | 12360 | 1.3125          | 1.3126 | 0.6881  | 0.7026    |
| 0.0678        | 31.0  | 12772 | 1.3224          | 1.3224 | 0.6780  | 0.6914    |
| 0.062         | 32.0  | 13184 | 1.3932          | 1.3933 | 0.6925  | 0.7063    |
| 0.0561        | 33.0  | 13596 | 1.3001          | 1.3012 | 0.7078  | 0.7156    |
| 0.0521        | 34.0  | 14008 | 1.2374          | 1.2382 | 0.6980  | 0.7127    |
| 0.0513        | 35.0  | 14420 | 1.2457          | 1.2461 | 0.6907  | 0.7126    |
| 0.0475        | 36.0  | 14832 | 1.2203          | 1.2183 | 0.7171  | 0.7293    |
| 0.0468        | 37.0  | 15244 | 1.1596          | 1.1585 | 0.7090  | 0.7238    |
| 0.0423        | 38.0  | 15656 | 1.1950          | 1.1953 | 0.7156  | 0.7317    |
| 0.0413        | 39.0  | 16068 | 1.1832          | 1.1835 | 0.7121  | 0.7260    |
| 0.0384        | 40.0  | 16480 | 1.2525          | 1.2535 | 0.7030  | 0.7149    |


### Framework versions

- Transformers 4.50.0.dev0
- Pytorch 2.6.0+cu118
- Datasets 3.3.2
- Tokenizers 0.21.0
