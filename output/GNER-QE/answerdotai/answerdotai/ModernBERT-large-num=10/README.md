---
library_name: transformers
license: apache-2.0
base_model: answerdotai/ModernBERT-large
tags:
- generated_from_trainer
metrics:
- spearmanr
model-index:
- name: ModernBERT-large-num=10
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# ModernBERT-large-num=10

This model is a fine-tuned version of [answerdotai/ModernBERT-large](https://huggingface.co/answerdotai/ModernBERT-large) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.9744
- Mse: 0.9740
- Pearson: 0.7776
- Spearmanr: 0.7848

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
- gradient_accumulation_steps: 4
- total_train_batch_size: 16
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: linear
- num_epochs: 40.0

### Training results

| Training Loss | Epoch   | Step  | Validation Loss | Mse    | Pearson | Spearmanr |
|:-------------:|:-------:|:-----:|:---------------:|:------:|:-------:|:---------:|
| 5.5301        | 0.9991  | 823   | 1.3608          | 1.3604 | 0.6322  | 0.6497    |
| 3.2442        | 1.9991  | 1646  | 1.2214          | 1.2209 | 0.6685  | 0.6738    |
| 2.7996        | 2.9991  | 2469  | 1.3547          | 1.3546 | 0.6601  | 0.6769    |
| 2.6997        | 3.9991  | 3292  | 1.4101          | 1.4100 | 0.6653  | 0.6716    |
| 2.0402        | 4.9991  | 4115  | 1.1748          | 1.1753 | 0.6890  | 0.6966    |
| 1.4786        | 5.9991  | 4938  | 1.2295          | 1.2298 | 0.7218  | 0.7307    |
| 1.264         | 6.9991  | 5761  | 1.1355          | 1.1361 | 0.6979  | 0.7037    |
| 1.3273        | 7.9991  | 6584  | 1.1697          | 1.1693 | 0.7281  | 0.7371    |
| 0.9419        | 8.9991  | 7407  | 1.1491          | 1.1493 | 0.7207  | 0.7259    |
| 0.6421        | 9.9991  | 8230  | 0.9538          | 0.9533 | 0.7524  | 0.7566    |
| 0.5874        | 10.9991 | 9053  | 1.1805          | 1.1803 | 0.7167  | 0.7204    |
| 0.6463        | 11.9991 | 9876  | 1.0785          | 1.0786 | 0.7306  | 0.7390    |
| 0.4513        | 12.9991 | 10699 | 1.0666          | 1.0667 | 0.7388  | 0.7479    |
| 0.3228        | 13.9991 | 11522 | 1.1567          | 1.1562 | 0.7259  | 0.7412    |
| 0.3078        | 14.9991 | 12345 | 1.0510          | 1.0510 | 0.7307  | 0.7450    |
| 0.4023        | 15.9991 | 13168 | 1.1633          | 1.1641 | 0.7122  | 0.7262    |
| 0.2707        | 16.9991 | 13991 | 1.0592          | 1.0594 | 0.7375  | 0.7468    |
| 0.1753        | 17.9991 | 14814 | 1.0760          | 1.0758 | 0.7272  | 0.7355    |
| 0.1818        | 18.9991 | 15637 | 1.1353          | 1.1355 | 0.7129  | 0.7246    |
| 0.2389        | 19.9991 | 16460 | 0.9744          | 0.9740 | 0.7776  | 0.7848    |
| 0.1807        | 20.9991 | 17283 | 1.0754          | 1.0749 | 0.7524  | 0.7695    |
| 0.1339        | 21.9991 | 18106 | 1.0387          | 1.0388 | 0.7584  | 0.7735    |
| 0.126         | 22.9991 | 18929 | 1.1348          | 1.1352 | 0.7221  | 0.7399    |
| 0.1819        | 23.9991 | 19752 | 1.1599          | 1.1599 | 0.7148  | 0.7298    |
| 0.1299        | 24.9991 | 20575 | 1.2343          | 1.2339 | 0.7003  | 0.7187    |
| 0.0891        | 25.9991 | 21398 | 1.1276          | 1.1273 | 0.7278  | 0.7451    |
| 0.1016        | 26.9991 | 22221 | 1.0969          | 1.0969 | 0.7293  | 0.7438    |
| 0.1399        | 27.9991 | 23044 | 1.1925          | 1.1933 | 0.7105  | 0.7248    |
| 0.1031        | 28.9991 | 23867 | 0.9824          | 0.9819 | 0.7576  | 0.7718    |
| 0.0948        | 29.9991 | 24690 | 1.0158          | 1.0158 | 0.7534  | 0.7686    |
| 0.0836        | 30.9991 | 25513 | 1.1269          | 1.1273 | 0.7416  | 0.7597    |
| 0.1042        | 31.9991 | 26336 | 1.1452          | 1.1457 | 0.7370  | 0.7561    |
| 0.0796        | 32.9991 | 27159 | 1.1698          | 1.1692 | 0.7295  | 0.7495    |
| 0.0575        | 33.9991 | 27982 | 1.1900          | 1.1901 | 0.7144  | 0.7325    |
| 0.0694        | 34.9991 | 28805 | 1.1226          | 1.1224 | 0.7360  | 0.7497    |
| 0.1234        | 35.9991 | 29628 | 1.1834          | 1.1831 | 0.7084  | 0.7273    |
| 0.1106        | 36.9991 | 30451 | 1.3435          | 1.3428 | 0.7089  | 0.7282    |
| 0.0899        | 37.9991 | 31274 | 1.1488          | 1.1488 | 0.7219  | 0.7354    |
| 0.092         | 38.9991 | 32097 | 1.2724          | 1.2717 | 0.7114  | 0.7272    |
| 0.0874        | 39.9991 | 32920 | 1.1692          | 1.1690 | 0.7242  | 0.7481    |


### Framework versions

- Transformers 4.50.0.dev0
- Pytorch 2.6.0+cu118
- Datasets 3.3.2
- Tokenizers 0.21.0
