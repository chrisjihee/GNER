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
- name: deberta-v3-large-ep1-lr1e-5
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
      value: 0.5963151463076831
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# deberta-v3-large-ep1-lr1e-5

This model is a fine-tuned version of [microsoft/deberta-v3-large](https://huggingface.co/microsoft/deberta-v3-large) on the GLUE ZSE-TEST-SAMPLED-N700-QUALITY_EST-MAX_SAMPLED=0 dataset.
It achieves the following results on the evaluation set:
- Loss: 2.2260
- Mse: 2.2261
- Pearson: 0.5955
- Spearmanr: 0.5963

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
- training_steps: 7000

### Training results

| Training Loss | Epoch  | Step | Validation Loss | Mse    | Pearson | Spearmanr |
|:-------------:|:------:|:----:|:---------------:|:------:|:-------:|:---------:|
| 1.0672        | 0.0397 | 500  | 2.4418          | 2.4416 | 0.5058  | 0.5169    |
| 0.9178        | 0.0794 | 1000 | 2.4476          | 2.4476 | 0.5454  | 0.5475    |
| 0.8805        | 0.1191 | 1500 | 2.3803          | 2.3809 | 0.5600  | 0.5630    |
| 0.7996        | 0.1589 | 2000 | 2.2375          | 2.2375 | 0.5564  | 0.5607    |
| 0.7314        | 0.1986 | 2500 | 2.4863          | 2.4859 | 0.5572  | 0.5542    |
| 0.6723        | 0.2383 | 3000 | 1.7254          | 1.7251 | 0.5727  | 0.5722    |
| 0.6505        | 0.2780 | 3500 | 2.2192          | 2.2189 | 0.5726  | 0.5674    |
| 0.6227        | 0.3177 | 4000 | 2.1712          | 2.1709 | 0.5830  | 0.5784    |
| 0.607         | 0.3574 | 4500 | 2.2976          | 2.2976 | 0.5775  | 0.5761    |
| 0.5042        | 0.3971 | 5000 | 1.8176          | 1.8174 | 0.5607  | 0.5589    |
| 0.5195        | 0.4369 | 5500 | 1.7324          | 1.7324 | 0.5764  | 0.5731    |
| 0.5091        | 0.4766 | 6000 | 2.2260          | 2.2261 | 0.5955  | 0.5963    |
| 0.4613        | 0.5163 | 6500 | 1.5301          | 1.5300 | 0.5790  | 0.5742    |
| 0.4126        | 0.5560 | 7000 | 1.7115          | 1.7113 | 0.5876  | 0.5845    |


### Framework versions

- Transformers 4.50.0.dev0
- Pytorch 2.6.0+cu118
- Datasets 3.4.1
- Tokenizers 0.21.1
