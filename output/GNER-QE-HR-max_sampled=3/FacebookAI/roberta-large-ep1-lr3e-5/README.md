---
library_name: transformers
language:
- en
license: mit
base_model: FacebookAI/roberta-large
tags:
- generated_from_trainer
datasets:
- glue
metrics:
- spearmanr
model-index:
- name: roberta-large-ep1-lr3e-5
  results:
  - task:
      name: Text Classification
      type: text-classification
    dataset:
      name: GLUE ZSE-TEST-SAMPLED-N700-QUALITY_EST-MAX_SAMPLED=0
      type: glue
      args: ZSE-test-sampled-N700-quality_est-max_sampled=0
    metrics:
    - name: Spearmanr
      type: spearmanr
      value: 0.4790225760486888
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# roberta-large-ep1-lr3e-5

This model is a fine-tuned version of [FacebookAI/roberta-large](https://huggingface.co/FacebookAI/roberta-large) on the GLUE ZSE-TEST-SAMPLED-N700-QUALITY_EST-MAX_SAMPLED=0 dataset.
It achieves the following results on the evaluation set:
- Loss: 2.6004
- Mse: 2.5992
- Pearson: 0.4670
- Spearmanr: 0.4790

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 3e-05
- train_batch_size: 16
- eval_batch_size: 30
- seed: 42
- distributed_type: multi-GPU
- num_devices: 4
- total_train_batch_size: 64
- total_eval_batch_size: 120
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- num_epochs: 1.0

### Training results

| Training Loss | Epoch  | Step  | Validation Loss | Mse    | Pearson | Spearmanr |
|:-------------:|:------:|:-----:|:---------------:|:------:|:-------:|:---------:|
| 1.2167        | 0.0397 | 500   | 2.9386          | 2.9384 | 0.4194  | 0.4288    |
| 1.1519        | 0.0794 | 1000  | 2.0107          | 2.0107 | 0.4290  | 0.4311    |
| 1.1078        | 0.1191 | 1500  | 2.6813          | 2.6809 | 0.4441  | 0.4485    |
| 1.0832        | 0.1589 | 2000  | 2.1238          | 2.1239 | 0.4544  | 0.4601    |
| 1.0952        | 0.1986 | 2500  | 2.2721          | 2.2722 | 0.4062  | 0.4103    |
| 1.0034        | 0.2383 | 3000  | 1.9691          | 1.9686 | 0.4643  | 0.4656    |
| 1.0175        | 0.2780 | 3500  | 2.6004          | 2.5992 | 0.4670  | 0.4790    |
| 0.9828        | 0.3177 | 4000  | 2.4547          | 2.4535 | 0.4192  | 0.4175    |
| 1.0097        | 0.3574 | 4500  | 3.0550          | 3.0541 | 0.4286  | 0.4284    |
| 0.8833        | 0.3971 | 5000  | 2.3181          | 2.3179 | 0.4545  | 0.4598    |
| 0.8887        | 0.4369 | 5500  | 2.6654          | 2.6650 | 0.4155  | 0.4204    |
| 0.9067        | 0.4766 | 6000  | 2.4933          | 2.4926 | 0.4260  | 0.4285    |
| 0.8597        | 0.5163 | 6500  | 2.1761          | 2.1759 | 0.4093  | 0.4209    |
| 0.8713        | 0.5560 | 7000  | 2.1295          | 2.1292 | 0.4126  | 0.4212    |
| 0.8733        | 0.5957 | 7500  | 2.4509          | 2.4508 | 0.4188  | 0.4197    |
| 0.8118        | 0.6354 | 8000  | 2.2136          | 2.2133 | 0.4075  | 0.4109    |
| 0.8434        | 0.6751 | 8500  | 2.6300          | 2.6295 | 0.3921  | 0.3837    |
| 0.8196        | 0.7149 | 9000  | 2.7535          | 2.7531 | 0.4432  | 0.4461    |
| 0.7692        | 0.7546 | 9500  | 2.5531          | 2.5525 | 0.4111  | 0.4101    |
| 0.7699        | 0.7943 | 10000 | 1.7374          | 1.7368 | 0.3759  | 0.3758    |
| 0.7732        | 0.8340 | 10500 | 2.3577          | 2.3570 | 0.4186  | 0.4281    |
| 0.7745        | 0.8737 | 11000 | 2.7739          | 2.7737 | 0.4026  | 0.4141    |
| 0.8576        | 0.9134 | 11500 | 1.8865          | 1.8862 | 0.3349  | 0.3462    |
| 0.8397        | 0.9531 | 12000 | 2.1265          | 2.1261 | 0.3887  | 0.3987    |
| 0.8963        | 0.9929 | 12500 | 2.8307          | 2.8298 | 0.3655  | 0.3904    |


### Framework versions

- Transformers 4.50.0.dev0
- Pytorch 2.6.0+cu118
- Datasets 3.4.1
- Tokenizers 0.21.1
