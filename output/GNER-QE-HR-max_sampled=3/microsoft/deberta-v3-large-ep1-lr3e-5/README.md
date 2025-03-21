---
library_name: transformers
language:
- en
license: mit
base_model: microsoft/deberta-v3-large
tags:
- generated_from_trainer
datasets:
- glue
metrics:
- spearmanr
model-index:
- name: deberta-v3-large-ep1-lr3e-5
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
      value: 0.5258787314087364
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# deberta-v3-large-ep1-lr3e-5

This model is a fine-tuned version of [microsoft/deberta-v3-large](https://huggingface.co/microsoft/deberta-v3-large) on the GLUE ZSE-TEST-SAMPLED-N700-QUALITY_EST-MAX_SAMPLED=0 dataset.
It achieves the following results on the evaluation set:
- Loss: 2.8636
- Mse: 2.8627
- Pearson: 0.5237
- Spearmanr: 0.5259

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
- training_steps: 7000

### Training results

| Training Loss | Epoch  | Step | Validation Loss | Mse    | Pearson | Spearmanr |
|:-------------:|:------:|:----:|:---------------:|:------:|:-------:|:---------:|
| 1.1002        | 0.0397 | 500  | 2.4354          | 2.4350 | 0.5115  | 0.5201    |
| 0.963         | 0.0794 | 1000 | 2.1048          | 2.1048 | 0.4523  | 0.4620    |
| 0.8677        | 0.1191 | 1500 | 2.2776          | 2.2773 | 0.4933  | 0.4978    |
| 0.7958        | 0.1589 | 2000 | 1.9427          | 1.9428 | 0.5032  | 0.5018    |
| 0.7146        | 0.1986 | 2500 | 2.7738          | 2.7736 | 0.4511  | 0.4536    |
| 0.6724        | 0.2383 | 3000 | 1.7077          | 1.7075 | 0.4654  | 0.4669    |
| 0.6289        | 0.2780 | 3500 | 2.0951          | 2.0947 | 0.4967  | 0.4901    |
| 0.6191        | 0.3177 | 4000 | 2.8636          | 2.8627 | 0.5237  | 0.5259    |
| 0.609         | 0.3574 | 4500 | 2.5508          | 2.5507 | 0.4848  | 0.4865    |
| 0.486         | 0.3971 | 5000 | 1.9257          | 1.9253 | 0.4308  | 0.4395    |
| 0.5011        | 0.4369 | 5500 | 2.4780          | 2.4776 | 0.4516  | 0.4504    |
| 0.4697        | 0.4766 | 6000 | 2.3036          | 2.3031 | 0.4017  | 0.4052    |
| 0.4372        | 0.5163 | 6500 | 1.8966          | 1.8962 | 0.4595  | 0.4661    |
| 0.4183        | 0.5560 | 7000 | 2.0740          | 2.0735 | 0.4725  | 0.4799    |


### Framework versions

- Transformers 4.50.0.dev0
- Pytorch 2.6.0+cu118
- Datasets 3.4.1
- Tokenizers 0.21.1
