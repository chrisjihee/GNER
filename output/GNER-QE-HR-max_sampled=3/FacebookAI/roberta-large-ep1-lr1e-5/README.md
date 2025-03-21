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
- name: roberta-large-ep1-lr1e-5
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
      value: 0.5390567981865767
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# roberta-large-ep1-lr1e-5

This model is a fine-tuned version of [FacebookAI/roberta-large](https://huggingface.co/FacebookAI/roberta-large) on the GLUE ZSE-TEST-SAMPLED-N700-QUALITY_EST-MAX_SAMPLED=0 dataset.
It achieves the following results on the evaluation set:
- Loss: 2.6020
- Mse: 2.6016
- Pearson: 0.5315
- Spearmanr: 0.5391

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 1e-05
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
| 1.1146        | 0.0397 | 500   | 2.5211          | 2.5211 | 0.4594  | 0.4675    |
| 1.0328        | 0.0794 | 1000  | 2.8020          | 2.8015 | 0.5117  | 0.5255    |
| 0.9755        | 0.1191 | 1500  | 2.6020          | 2.6016 | 0.5315  | 0.5391    |
| 0.9246        | 0.1589 | 2000  | 2.3082          | 2.3083 | 0.5264  | 0.5382    |
| 0.8917        | 0.1986 | 2500  | 2.4118          | 2.4118 | 0.4842  | 0.4855    |
| 0.8313        | 0.2383 | 3000  | 1.8066          | 1.8065 | 0.4931  | 0.4987    |
| 0.8184        | 0.2780 | 3500  | 2.2878          | 2.2873 | 0.5208  | 0.5185    |
| 0.7876        | 0.3177 | 4000  | 2.0631          | 2.0628 | 0.5046  | 0.5000    |
| 0.8           | 0.3574 | 4500  | 2.2693          | 2.2690 | 0.5155  | 0.5138    |
| 0.6893        | 0.3971 | 5000  | 1.9561          | 1.9562 | 0.4943  | 0.4952    |
| 0.6941        | 0.4369 | 5500  | 2.1833          | 2.1829 | 0.5150  | 0.5172    |
| 0.688         | 0.4766 | 6000  | 1.9256          | 1.9254 | 0.4780  | 0.4761    |
| 0.6487        | 0.5163 | 6500  | 1.6590          | 1.6588 | 0.4766  | 0.4879    |
| 0.5932        | 0.5560 | 7000  | 1.9415          | 1.9412 | 0.4904  | 0.4980    |
| 0.6144        | 0.5957 | 7500  | 2.0783          | 2.0780 | 0.4883  | 0.4880    |
| 0.5835        | 0.6354 | 8000  | 2.1246          | 2.1239 | 0.5124  | 0.5121    |
| 0.5863        | 0.6751 | 8500  | 2.1132          | 2.1130 | 0.4753  | 0.4860    |
| 0.5645        | 0.7149 | 9000  | 2.1345          | 2.1341 | 0.4960  | 0.4981    |
| 0.5523        | 0.7546 | 9500  | 2.2703          | 2.2699 | 0.4733  | 0.4798    |
| 0.5059        | 0.7943 | 10000 | 1.8604          | 1.8602 | 0.4487  | 0.4596    |
| 0.5054        | 0.8340 | 10500 | 2.1557          | 2.1552 | 0.4778  | 0.4827    |
| 0.4985        | 0.8737 | 11000 | 2.1775          | 2.1773 | 0.4848  | 0.4897    |
| 0.4795        | 0.9134 | 11500 | 2.3048          | 2.3045 | 0.4895  | 0.4975    |
| 0.4869        | 0.9531 | 12000 | 1.9517          | 1.9515 | 0.5196  | 0.5262    |
| 0.4781        | 0.9929 | 12500 | 1.9705          | 1.9702 | 0.5012  | 0.5092    |


### Framework versions

- Transformers 4.50.0.dev0
- Pytorch 2.6.0+cu118
- Datasets 3.4.1
- Tokenizers 0.21.1
