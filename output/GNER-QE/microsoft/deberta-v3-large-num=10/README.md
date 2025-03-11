---
library_name: transformers
license: mit
base_model: microsoft/deberta-v3-large
tags:
- generated_from_trainer
metrics:
- spearmanr
model-index:
- name: deberta-v3-large-num=10
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# deberta-v3-large-num=10

This model is a fine-tuned version of [microsoft/deberta-v3-large](https://huggingface.co/microsoft/deberta-v3-large) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.8757
- Mse: 0.8765
- Pearson: 0.8209
- Spearmanr: 0.8352

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
| 1.3551        | 1.0   | 412   | 1.1470          | 1.1468 | 0.6888  | 0.6985    |
| 0.6693        | 2.0   | 824   | 0.9709          | 0.9698 | 0.7573  | 0.7717    |
| 0.4776        | 3.0   | 1236  | 0.9503          | 0.9506 | 0.7657  | 0.7826    |
| 0.3542        | 4.0   | 1648  | 0.8773          | 0.8775 | 0.7950  | 0.8015    |
| 0.2502        | 5.0   | 2060  | 1.0275          | 1.0266 | 0.7834  | 0.7952    |
| 0.1914        | 6.0   | 2472  | 0.9939          | 0.9936 | 0.7835  | 0.8026    |
| 0.1485        | 7.0   | 2884  | 0.9551          | 0.9556 | 0.7846  | 0.7971    |
| 0.1101        | 8.0   | 3296  | 1.0661          | 1.0668 | 0.7698  | 0.7859    |
| 0.0997        | 9.0   | 3708  | 0.8372          | 0.8378 | 0.8058  | 0.8232    |
| 0.0732        | 10.0  | 4120  | 0.9233          | 0.9226 | 0.7830  | 0.8026    |
| 0.0681        | 11.0  | 4532  | 0.9287          | 0.9296 | 0.7940  | 0.8147    |
| 0.0505        | 12.0  | 4944  | 0.8296          | 0.8309 | 0.8116  | 0.8287    |
| 0.0431        | 13.0  | 5356  | 0.9215          | 0.9224 | 0.8087  | 0.8275    |
| 0.0381        | 14.0  | 5768  | 0.9282          | 0.9292 | 0.7934  | 0.8124    |
| 0.0291        | 15.0  | 6180  | 0.9785          | 0.9794 | 0.7808  | 0.8006    |
| 0.0311        | 16.0  | 6592  | 1.1376          | 1.1385 | 0.7600  | 0.7805    |
| 0.0249        | 17.0  | 7004  | 1.0120          | 1.0123 | 0.7738  | 0.7917    |
| 0.0242        | 18.0  | 7416  | 0.9796          | 0.9802 | 0.7849  | 0.8034    |
| 0.022         | 19.0  | 7828  | 0.8757          | 0.8765 | 0.8209  | 0.8352    |
| 0.0303        | 20.0  | 8240  | 0.9333          | 0.9342 | 0.7897  | 0.8069    |
| 0.0281        | 21.0  | 8652  | 1.1178          | 1.1178 | 0.7645  | 0.7899    |
| 0.0243        | 22.0  | 9064  | 0.9606          | 0.9613 | 0.7927  | 0.8106    |
| 0.0226        | 23.0  | 9476  | 1.0193          | 1.0202 | 0.7769  | 0.8009    |
| 0.0201        | 24.0  | 9888  | 0.9175          | 0.9179 | 0.7984  | 0.8201    |
| 0.0163        | 25.0  | 10300 | 0.9162          | 0.9164 | 0.7964  | 0.8174    |
| 0.0138        | 26.0  | 10712 | 1.0848          | 1.0850 | 0.7624  | 0.7868    |
| 0.0139        | 27.0  | 11124 | 1.0764          | 1.0772 | 0.7708  | 0.7913    |
| 0.0161        | 28.0  | 11536 | 0.8683          | 0.8685 | 0.8064  | 0.8285    |
| 0.0165        | 29.0  | 11948 | 0.9143          | 0.9151 | 0.7945  | 0.8179    |
| 0.0173        | 30.0  | 12360 | 1.0316          | 1.0329 | 0.7789  | 0.8019    |
| 0.0153        | 31.0  | 12772 | 0.9544          | 0.9547 | 0.7894  | 0.8071    |
| 0.0131        | 32.0  | 13184 | 0.9673          | 0.9682 | 0.7933  | 0.8120    |
| 0.0147        | 33.0  | 13596 | 0.9963          | 0.9973 | 0.7802  | 0.8058    |
| 0.0216        | 34.0  | 14008 | 0.9980          | 0.9984 | 0.7755  | 0.8028    |
| 0.0215        | 35.0  | 14420 | 0.9407          | 0.9412 | 0.7862  | 0.8063    |
| 0.0147        | 36.0  | 14832 | 1.0652          | 1.0659 | 0.7760  | 0.7964    |
| 0.0109        | 37.0  | 15244 | 1.0641          | 1.0651 | 0.7651  | 0.7881    |
| 0.0122        | 38.0  | 15656 | 0.9557          | 0.9565 | 0.7902  | 0.8119    |
| 0.0171        | 39.0  | 16068 | 1.1248          | 1.1263 | 0.7654  | 0.7913    |
| 0.014         | 40.0  | 16480 | 1.0761          | 1.0764 | 0.7603  | 0.7865    |


### Framework versions

- Transformers 4.50.0.dev0
- Pytorch 2.6.0+cu118
- Datasets 3.3.2
- Tokenizers 0.21.0
