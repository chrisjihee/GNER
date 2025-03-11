---
library_name: transformers
license: mit
base_model: microsoft/deberta-v3-base
tags:
- generated_from_trainer
metrics:
- spearmanr
model-index:
- name: deberta-v3-base-num=20
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# deberta-v3-base-num=20

This model is a fine-tuned version of [microsoft/deberta-v3-base](https://huggingface.co/microsoft/deberta-v3-base) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.8694
- Mse: 0.8692
- Pearson: 0.7971
- Spearmanr: 0.8107

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
| 1.2012        | 0.9994  | 820   | 1.2298          | 1.2296 | 0.6665  | 0.6800    |
| 0.6865        | 1.9994  | 1640  | 1.1029          | 1.1028 | 0.7007  | 0.7087    |
| 0.5527        | 2.9994  | 2460  | 1.0211          | 1.0207 | 0.7232  | 0.7365    |
| 0.5155        | 3.9994  | 3280  | 1.2345          | 1.2350 | 0.6953  | 0.7095    |
| 0.3843        | 4.9994  | 4100  | 0.9180          | 0.9181 | 0.7538  | 0.7603    |
| 0.2967        | 5.9994  | 4920  | 1.0042          | 1.0039 | 0.7467  | 0.7590    |
| 0.255         | 6.9994  | 5740  | 1.0244          | 1.0237 | 0.7587  | 0.7744    |
| 0.2576        | 7.9994  | 6560  | 1.1344          | 1.1345 | 0.7481  | 0.7603    |
| 0.1923        | 8.9994  | 7380  | 1.0536          | 1.0531 | 0.7363  | 0.7526    |
| 0.1416        | 9.9994  | 8200  | 0.9227          | 0.9225 | 0.7734  | 0.8014    |
| 0.1242        | 10.9994 | 9020  | 0.9219          | 0.9217 | 0.7926  | 0.8034    |
| 0.1267        | 11.9994 | 9840  | 0.9961          | 0.9959 | 0.7527  | 0.7672    |
| 0.1           | 12.9994 | 10660 | 0.8613          | 0.8607 | 0.7990  | 0.8167    |
| 0.0741        | 13.9994 | 11480 | 0.8968          | 0.8968 | 0.7907  | 0.8067    |
| 0.0712        | 14.9994 | 12300 | 0.9104          | 0.9102 | 0.7860  | 0.8050    |
| 0.0733        | 15.9994 | 13120 | 0.8131          | 0.8133 | 0.7966  | 0.8184    |
| 0.06          | 16.9994 | 13940 | 0.8469          | 0.8469 | 0.7928  | 0.8066    |
| 0.0499        | 17.9994 | 14760 | 0.8487          | 0.8490 | 0.7945  | 0.8090    |
| 0.0487        | 18.9994 | 15580 | 0.9634          | 0.9629 | 0.7729  | 0.7946    |
| 0.0471        | 19.9994 | 16400 | 0.9059          | 0.9059 | 0.7801  | 0.7970    |
| 0.0424        | 20.9994 | 17220 | 0.8508          | 0.8509 | 0.8172  | 0.8274    |
| 0.0332        | 21.9994 | 18040 | 0.9516          | 0.9514 | 0.7689  | 0.7885    |
| 0.0339        | 22.9994 | 18860 | 0.9923          | 0.9923 | 0.7589  | 0.7749    |
| 0.0355        | 23.9994 | 19680 | 0.8614          | 0.8609 | 0.8006  | 0.8172    |
| 0.0326        | 24.9994 | 20500 | 0.8509          | 0.8506 | 0.7993  | 0.8153    |
| 0.0267        | 25.9994 | 21320 | 0.9858          | 0.9855 | 0.7583  | 0.7667    |
| 0.0289        | 26.9994 | 22140 | 0.9934          | 0.9935 | 0.7716  | 0.7861    |
| 0.0317        | 27.9994 | 22960 | 0.9511          | 0.9510 | 0.7801  | 0.8001    |
| 0.0303        | 28.9994 | 23780 | 0.9271          | 0.9268 | 0.7899  | 0.8047    |
| 0.0205        | 29.9994 | 24600 | 0.9080          | 0.9079 | 0.7787  | 0.7933    |
| 0.0219        | 30.9994 | 25420 | 1.0110          | 1.0107 | 0.7530  | 0.7842    |
| 0.0245        | 31.9994 | 26240 | 0.9714          | 0.9712 | 0.7605  | 0.7774    |
| 0.0228        | 32.9994 | 27060 | 0.9140          | 0.9139 | 0.7890  | 0.8051    |
| 0.0169        | 33.9994 | 27880 | 0.8703          | 0.8702 | 0.7971  | 0.8135    |
| 0.0168        | 34.9994 | 28700 | 0.9433          | 0.9430 | 0.7715  | 0.7850    |
| 0.0173        | 35.9994 | 29520 | 1.0085          | 1.0086 | 0.7550  | 0.7742    |
| 0.0214        | 36.9994 | 30340 | 0.8715          | 0.8708 | 0.8057  | 0.8207    |
| 0.0144        | 37.9994 | 31160 | 0.9940          | 0.9934 | 0.7779  | 0.7961    |
| 0.0163        | 38.9994 | 31980 | 0.8807          | 0.8807 | 0.7844  | 0.8063    |
| 0.0179        | 39.9994 | 32800 | 0.8694          | 0.8692 | 0.7971  | 0.8107    |


### Framework versions

- Transformers 4.50.0.dev0
- Pytorch 2.6.0+cu118
- Datasets 3.3.2
- Tokenizers 0.21.0
