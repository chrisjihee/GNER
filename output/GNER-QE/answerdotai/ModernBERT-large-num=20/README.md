---
library_name: transformers
license: apache-2.0
base_model: answerdotai/ModernBERT-large
tags:
- generated_from_trainer
metrics:
- spearmanr
model-index:
- name: ModernBERT-large-num=20
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# ModernBERT-large-num=20

This model is a fine-tuned version of [answerdotai/ModernBERT-large](https://huggingface.co/answerdotai/ModernBERT-large) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.7490
- Mse: 0.7491
- Pearson: 0.8254
- Spearmanr: 0.8401

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

| Training Loss | Epoch | Step  | Validation Loss | Mse    | Pearson | Spearmanr |
|:-------------:|:-----:|:-----:|:---------------:|:------:|:-------:|:---------:|
| 4.0827        | 1.0   | 1641  | 1.2063          | 1.2061 | 0.6578  | 0.6746    |
| 1.9578        | 2.0   | 3282  | 0.9705          | 0.9708 | 0.7488  | 0.7544    |
| 0.9816        | 3.0   | 4923  | 0.9104          | 0.9097 | 0.7884  | 0.7942    |
| 0.5643        | 4.0   | 6564  | 0.8375          | 0.8374 | 0.7969  | 0.8082    |
| 0.3501        | 5.0   | 8205  | 0.9122          | 0.9124 | 0.7661  | 0.7758    |
| 0.2411        | 6.0   | 9846  | 0.8382          | 0.8377 | 0.7945  | 0.8050    |
| 0.185         | 7.0   | 11487 | 0.7977          | 0.7975 | 0.7968  | 0.8098    |
| 0.1461        | 8.0   | 13128 | 0.9597          | 0.9597 | 0.7640  | 0.7735    |
| 0.1279        | 9.0   | 14769 | 0.9117          | 0.9113 | 0.7891  | 0.7993    |
| 0.1088        | 10.0  | 16410 | 0.9695          | 0.9690 | 0.7579  | 0.7639    |
| 0.1016        | 11.0  | 18051 | 0.8474          | 0.8476 | 0.7853  | 0.7958    |
| 0.0757        | 12.0  | 19692 | 0.9031          | 0.9030 | 0.7696  | 0.7822    |
| 0.0767        | 13.0  | 21333 | 0.9837          | 0.9834 | 0.7485  | 0.7613    |
| 0.0626        | 14.0  | 22974 | 0.9169          | 0.9169 | 0.7700  | 0.7795    |
| 0.069         | 15.0  | 24615 | 1.0208          | 1.0208 | 0.7539  | 0.7669    |
| 0.05          | 16.0  | 26256 | 0.9534          | 0.9535 | 0.7673  | 0.7828    |
| 0.0468        | 17.0  | 27897 | 0.9644          | 0.9644 | 0.7606  | 0.7779    |
| 0.0563        | 18.0  | 29538 | 1.0149          | 1.0147 | 0.7438  | 0.7605    |
| 0.0776        | 19.0  | 31179 | 1.0172          | 1.0171 | 0.7574  | 0.7681    |
| 0.0863        | 20.0  | 32820 | 0.9337          | 0.9338 | 0.7754  | 0.7904    |
| 0.0671        | 21.0  | 34461 | 0.7913          | 0.7912 | 0.8040  | 0.8141    |
| 0.0446        | 22.0  | 36102 | 0.9107          | 0.9104 | 0.7758  | 0.7904    |
| 0.042         | 23.0  | 37743 | 0.9151          | 0.9145 | 0.7782  | 0.7892    |
| 0.0382        | 24.0  | 39384 | 0.9035          | 0.9037 | 0.7783  | 0.7920    |
| 0.0355        | 25.0  | 41025 | 0.8374          | 0.8372 | 0.7963  | 0.8150    |
| 0.0527        | 26.0  | 42666 | 0.8390          | 0.8394 | 0.7985  | 0.8114    |
| 0.0511        | 27.0  | 44307 | 1.0080          | 1.0078 | 0.7702  | 0.7814    |
| 0.0431        | 28.0  | 45948 | 0.8801          | 0.8801 | 0.7845  | 0.8017    |
| 0.0414        | 29.0  | 47589 | 0.8483          | 0.8478 | 0.8126  | 0.8253    |
| 0.0296        | 30.0  | 49230 | 0.8572          | 0.8573 | 0.7942  | 0.8098    |
| 0.0488        | 31.0  | 50871 | 0.7312          | 0.7315 | 0.8223  | 0.8354    |
| 0.0373        | 32.0  | 52512 | 0.8897          | 0.8892 | 0.7872  | 0.8034    |
| 0.0449        | 33.0  | 54153 | 0.7490          | 0.7491 | 0.8254  | 0.8401    |
| 0.0297        | 34.0  | 55794 | 0.7816          | 0.7816 | 0.8046  | 0.8204    |
| 0.0302        | 35.0  | 57435 | 0.8581          | 0.8579 | 0.7870  | 0.8026    |
| 0.0283        | 36.0  | 59076 | 0.9137          | 0.9129 | 0.7833  | 0.8004    |
| 0.0259        | 37.0  | 60717 | 0.8642          | 0.8639 | 0.7960  | 0.8114    |
| 0.0234        | 38.0  | 62358 | 0.9008          | 0.9007 | 0.7855  | 0.8035    |
| 0.0167        | 39.0  | 63999 | 0.9492          | 0.9489 | 0.7677  | 0.7873    |
| 0.0197        | 40.0  | 65640 | 0.9499          | 0.9499 | 0.7827  | 0.7965    |


### Framework versions

- Transformers 4.50.0.dev0
- Pytorch 2.6.0+cu118
- Datasets 3.3.2
- Tokenizers 0.21.0
