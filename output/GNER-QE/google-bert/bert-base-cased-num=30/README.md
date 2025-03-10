---
library_name: transformers
license: apache-2.0
base_model: google-bert/bert-base-cased
tags:
- generated_from_trainer
metrics:
- spearmanr
model-index:
- name: bert-base-cased-num=30
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# bert-base-cased-num=30

This model is a fine-tuned version of [google-bert/bert-base-cased](https://huggingface.co/google-bert/bert-base-cased) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.9574
- Mse: 0.9572
- Pearson: 0.7679
- Spearmanr: 0.7800

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
| 1.0101        | 0.9994  | 1227  | 1.1701          | 1.1701 | 0.6496  | 0.6668    |
| 0.6721        | 1.9994  | 2454  | 1.0345          | 1.0344 | 0.6997  | 0.7117    |
| 0.4873        | 2.9994  | 3681  | 0.8998          | 0.8995 | 0.7472  | 0.7569    |
| 0.3946        | 3.9994  | 4908  | 1.0887          | 1.0888 | 0.7306  | 0.7433    |
| 0.2969        | 4.9994  | 6135  | 0.9259          | 0.9254 | 0.7452  | 0.7529    |
| 0.2491        | 5.9994  | 7362  | 0.9333          | 0.9333 | 0.7421  | 0.7487    |
| 0.1861        | 6.9994  | 8589  | 0.9255          | 0.9253 | 0.7563  | 0.7646    |
| 0.1554        | 7.9994  | 9816  | 0.9131          | 0.9131 | 0.7452  | 0.7625    |
| 0.1173        | 8.9994  | 11043 | 0.9806          | 0.9808 | 0.7504  | 0.7638    |
| 0.1005        | 9.9994  | 12270 | 0.8897          | 0.8896 | 0.7554  | 0.7740    |
| 0.0766        | 10.9994 | 13497 | 0.9002          | 0.9002 | 0.7660  | 0.7781    |
| 0.0682        | 11.9994 | 14724 | 0.8522          | 0.8520 | 0.7695  | 0.7879    |
| 0.0543        | 12.9994 | 15951 | 0.9518          | 0.9519 | 0.7484  | 0.7654    |
| 0.0477        | 13.9994 | 17178 | 1.0226          | 1.0223 | 0.7297  | 0.7517    |
| 0.0424        | 14.9994 | 18405 | 0.8491          | 0.8491 | 0.7771  | 0.7917    |
| 0.0411        | 15.9994 | 19632 | 0.9577          | 0.9578 | 0.7502  | 0.7748    |
| 0.0333        | 16.9994 | 20859 | 0.8653          | 0.8651 | 0.7704  | 0.7880    |
| 0.0327        | 17.9994 | 22086 | 0.9143          | 0.9142 | 0.7639  | 0.7794    |
| 0.0294        | 18.9994 | 23313 | 0.9643          | 0.9643 | 0.7594  | 0.7729    |
| 0.0292        | 19.9994 | 24540 | 0.8666          | 0.8666 | 0.7811  | 0.7967    |
| 0.0256        | 20.9994 | 25767 | 0.8672          | 0.8671 | 0.7799  | 0.7922    |
| 0.0277        | 21.9994 | 26994 | 0.8327          | 0.8327 | 0.7859  | 0.8001    |
| 0.022         | 22.9994 | 28221 | 0.8610          | 0.8611 | 0.7809  | 0.7960    |
| 0.0255        | 23.9994 | 29448 | 0.8569          | 0.8568 | 0.7947  | 0.8081    |
| 0.021         | 24.9994 | 30675 | 0.9878          | 0.9881 | 0.7472  | 0.7667    |
| 0.0215        | 25.9994 | 31902 | 1.0301          | 1.0304 | 0.7300  | 0.7507    |
| 0.02          | 26.9994 | 33129 | 0.9056          | 0.9055 | 0.7785  | 0.7964    |
| 0.0179        | 27.9994 | 34356 | 1.0594          | 1.0593 | 0.7339  | 0.7559    |
| 0.0191        | 28.9994 | 35583 | 0.9825          | 0.9823 | 0.7456  | 0.7657    |
| 0.0157        | 29.9994 | 36810 | 1.1165          | 1.1167 | 0.7181  | 0.7365    |
| 0.0171        | 30.9994 | 38037 | 0.9784          | 0.9779 | 0.7589  | 0.7744    |
| 0.017         | 31.9994 | 39264 | 1.0177          | 1.0181 | 0.7526  | 0.7642    |
| 0.0183        | 32.9994 | 40491 | 0.9761          | 0.9762 | 0.7543  | 0.7725    |
| 0.0149        | 33.9994 | 41718 | 0.8396          | 0.8394 | 0.7798  | 0.7955    |
| 0.0147        | 34.9994 | 42945 | 1.0561          | 1.0564 | 0.7350  | 0.7543    |
| 0.0147        | 35.9994 | 44172 | 0.9572          | 0.9573 | 0.7727  | 0.7877    |
| 0.0152        | 36.9994 | 45399 | 0.8817          | 0.8818 | 0.7793  | 0.7849    |
| 0.0119        | 37.9994 | 46626 | 0.9479          | 0.9480 | 0.7697  | 0.7812    |
| 0.0136        | 38.9994 | 47853 | 0.9597          | 0.9596 | 0.7523  | 0.7671    |
| 0.0123        | 39.9994 | 49080 | 0.9574          | 0.9572 | 0.7679  | 0.7800    |


### Framework versions

- Transformers 4.50.0.dev0
- Pytorch 2.6.0+cu118
- Datasets 3.3.2
- Tokenizers 0.21.0
