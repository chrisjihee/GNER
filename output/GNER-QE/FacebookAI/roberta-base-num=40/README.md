---
library_name: transformers
license: mit
base_model: FacebookAI/roberta-base
tags:
- generated_from_trainer
metrics:
- spearmanr
model-index:
- name: roberta-base-num=40
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# roberta-base-num=40

This model is a fine-tuned version of [FacebookAI/roberta-base](https://huggingface.co/FacebookAI/roberta-base) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.9158
- Mse: 0.9158
- Pearson: 0.7551
- Spearmanr: 0.7639

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
| 1.0049        | 1.0   | 1634  | 1.1640          | 1.1640 | 0.6565  | 0.6629    |
| 0.6475        | 2.0   | 3268  | 1.4085          | 1.4082 | 0.6518  | 0.6494    |
| 0.5236        | 3.0   | 4902  | 1.2493          | 1.2499 | 0.6419  | 0.6435    |
| 0.4312        | 4.0   | 6536  | 1.2097          | 1.2094 | 0.6453  | 0.6490    |
| 0.3666        | 5.0   | 8170  | 1.3148          | 1.3148 | 0.6351  | 0.6492    |
| 0.3158        | 6.0   | 9804  | 1.2454          | 1.2459 | 0.6489  | 0.6725    |
| 0.2695        | 7.0   | 11438 | 1.2065          | 1.2063 | 0.6849  | 0.6922    |
| 0.2306        | 8.0   | 13072 | 1.0450          | 1.0448 | 0.7012  | 0.7134    |
| 0.1959        | 9.0   | 14706 | 1.0675          | 1.0679 | 0.7194  | 0.7269    |
| 0.1676        | 10.0  | 16340 | 0.9779          | 0.9778 | 0.7338  | 0.7460    |
| 0.1446        | 11.0  | 17974 | 0.9539          | 0.9537 | 0.7334  | 0.7430    |
| 0.1238        | 12.0  | 19608 | 0.9627          | 0.9624 | 0.7357  | 0.7497    |
| 0.1054        | 13.0  | 21242 | 1.0524          | 1.0525 | 0.7153  | 0.7280    |
| 0.0903        | 14.0  | 22876 | 1.0049          | 1.0047 | 0.7240  | 0.7413    |
| 0.0777        | 15.0  | 24510 | 1.0269          | 1.0270 | 0.7276  | 0.7390    |
| 0.0685        | 16.0  | 26144 | 1.1125          | 1.1125 | 0.7068  | 0.7199    |
| 0.0584        | 17.0  | 27778 | 1.0544          | 1.0540 | 0.7096  | 0.7235    |
| 0.0545        | 18.0  | 29412 | 1.0295          | 1.0292 | 0.7227  | 0.7387    |
| 0.0456        | 19.0  | 31046 | 0.9736          | 0.9737 | 0.7442  | 0.7517    |
| 0.0397        | 20.0  | 32680 | 1.0324          | 1.0323 | 0.7184  | 0.7313    |
| 0.0376        | 21.0  | 34314 | 0.9505          | 0.9503 | 0.7386  | 0.7545    |
| 0.0326        | 22.0  | 35948 | 1.0003          | 1.0005 | 0.7344  | 0.7429    |
| 0.0302        | 23.0  | 37582 | 1.0182          | 1.0182 | 0.7310  | 0.7374    |
| 0.027         | 24.0  | 39216 | 0.9108          | 0.9110 | 0.7567  | 0.7696    |
| 0.0253        | 25.0  | 40850 | 0.9255          | 0.9257 | 0.7586  | 0.7703    |
| 0.0224        | 26.0  | 42484 | 0.9791          | 0.9790 | 0.7332  | 0.7439    |
| 0.0208        | 27.0  | 44118 | 0.9911          | 0.9915 | 0.7347  | 0.7442    |
| 0.0202        | 28.0  | 45752 | 0.9746          | 0.9749 | 0.7374  | 0.7552    |
| 0.0201        | 29.0  | 47386 | 1.0179          | 1.0177 | 0.7337  | 0.7389    |
| 0.0177        | 30.0  | 49020 | 1.0195          | 1.0194 | 0.7417  | 0.7486    |
| 0.0166        | 31.0  | 50654 | 0.9790          | 0.9790 | 0.7429  | 0.7614    |
| 0.0162        | 32.0  | 52288 | 0.9759          | 0.9757 | 0.7536  | 0.7616    |
| 0.0143        | 33.0  | 53922 | 0.9453          | 0.9453 | 0.7505  | 0.7611    |
| 0.0162        | 34.0  | 55556 | 0.9265          | 0.9265 | 0.7482  | 0.7546    |
| 0.0148        | 35.0  | 57190 | 0.9409          | 0.9404 | 0.7508  | 0.7549    |
| 0.013         | 36.0  | 58824 | 0.8867          | 0.8865 | 0.7647  | 0.7697    |
| 0.0138        | 37.0  | 60458 | 0.9444          | 0.9442 | 0.7522  | 0.7626    |
| 0.0118        | 38.0  | 62092 | 0.9091          | 0.9093 | 0.7597  | 0.7639    |
| 0.0132        | 39.0  | 63726 | 0.9128          | 0.9128 | 0.7557  | 0.7694    |
| 0.0117        | 40.0  | 65360 | 0.9158          | 0.9158 | 0.7551  | 0.7639    |


### Framework versions

- Transformers 4.50.0.dev0
- Pytorch 2.6.0+cu118
- Datasets 3.3.2
- Tokenizers 0.21.0
