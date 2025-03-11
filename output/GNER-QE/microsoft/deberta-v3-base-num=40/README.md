---
library_name: transformers
license: mit
base_model: microsoft/deberta-v3-base
tags:
- generated_from_trainer
metrics:
- spearmanr
model-index:
- name: deberta-v3-base-num=40
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# deberta-v3-base-num=40

This model is a fine-tuned version of [microsoft/deberta-v3-base](https://huggingface.co/microsoft/deberta-v3-base) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.7379
- Mse: 0.7379
- Pearson: 0.8159
- Spearmanr: 0.8280

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
| 0.9177        | 1.0   | 1634  | 1.0129          | 1.0129 | 0.7203  | 0.7296    |
| 0.4332        | 2.0   | 3268  | 0.9425          | 0.9425 | 0.7360  | 0.7423    |
| 0.2667        | 3.0   | 4902  | 1.0848          | 1.0847 | 0.7260  | 0.7411    |
| 0.1645        | 4.0   | 6536  | 0.7880          | 0.7881 | 0.7869  | 0.8004    |
| 0.1106        | 5.0   | 8170  | 1.0866          | 1.0866 | 0.7645  | 0.7793    |
| 0.0827        | 6.0   | 9804  | 0.9996          | 0.9993 | 0.7602  | 0.7790    |
| 0.0635        | 7.0   | 11438 | 1.0559          | 1.0562 | 0.7852  | 0.7982    |
| 0.0482        | 8.0   | 13072 | 1.0289          | 1.0289 | 0.7745  | 0.7874    |
| 0.0399        | 9.0   | 14706 | 0.9221          | 0.9223 | 0.7806  | 0.8037    |
| 0.0309        | 10.0  | 16340 | 0.9855          | 0.9853 | 0.7657  | 0.7843    |
| 0.028         | 11.0  | 17974 | 0.8909          | 0.8910 | 0.7712  | 0.7911    |
| 0.0234        | 12.0  | 19608 | 0.8729          | 0.8728 | 0.7866  | 0.8006    |
| 0.0214        | 13.0  | 21242 | 1.0085          | 1.0082 | 0.7589  | 0.7741    |
| 0.0212        | 14.0  | 22876 | 0.9412          | 0.9410 | 0.7677  | 0.7826    |
| 0.0203        | 15.0  | 24510 | 0.9965          | 0.9962 | 0.7509  | 0.7699    |
| 0.0179        | 16.0  | 26144 | 0.8846          | 0.8842 | 0.7771  | 0.7892    |
| 0.016         | 17.0  | 27778 | 0.9317          | 0.9314 | 0.7817  | 0.7949    |
| 0.0169        | 18.0  | 29412 | 0.9301          | 0.9299 | 0.7757  | 0.7827    |
| 0.0135        | 19.0  | 31046 | 0.9060          | 0.9059 | 0.7710  | 0.7843    |
| 0.012         | 20.0  | 32680 | 0.9786          | 0.9780 | 0.7675  | 0.7776    |
| 0.0123        | 21.0  | 34314 | 1.1179          | 1.1178 | 0.7543  | 0.7633    |
| 0.0118        | 22.0  | 35948 | 0.9224          | 0.9222 | 0.7683  | 0.7843    |
| 0.013         | 23.0  | 37582 | 0.9498          | 0.9497 | 0.7652  | 0.7826    |
| 0.0092        | 24.0  | 39216 | 0.8859          | 0.8857 | 0.7929  | 0.8085    |
| 0.0099        | 25.0  | 40850 | 0.8226          | 0.8225 | 0.8040  | 0.8182    |
| 0.0091        | 26.0  | 42484 | 0.8434          | 0.8434 | 0.8077  | 0.8234    |
| 0.0091        | 27.0  | 44118 | 0.7379          | 0.7379 | 0.8159  | 0.8280    |
| 0.0073        | 28.0  | 45752 | 0.8454          | 0.8452 | 0.7893  | 0.8043    |
| 0.0089        | 29.0  | 47386 | 0.8114          | 0.8114 | 0.8003  | 0.8168    |
| 0.0091        | 30.0  | 49020 | 1.0132          | 1.0128 | 0.7483  | 0.7675    |
| 0.0063        | 31.0  | 50654 | 0.9396          | 0.9393 | 0.7740  | 0.7936    |
| 0.0073        | 32.0  | 52288 | 0.8741          | 0.8740 | 0.7911  | 0.8054    |
| 0.0071        | 33.0  | 53922 | 0.9750          | 0.9746 | 0.7647  | 0.7791    |
| 0.0065        | 34.0  | 55556 | 1.0365          | 1.0362 | 0.7410  | 0.7519    |
| 0.0061        | 35.0  | 57190 | 0.8609          | 0.8609 | 0.7851  | 0.8055    |
| 0.0061        | 36.0  | 58824 | 0.9795          | 0.9796 | 0.7631  | 0.7912    |
| 0.0064        | 37.0  | 60458 | 0.8981          | 0.8978 | 0.7929  | 0.8013    |
| 0.0073        | 38.0  | 62092 | 0.9472          | 0.9473 | 0.7872  | 0.7996    |
| 0.0054        | 39.0  | 63726 | 0.9121          | 0.9122 | 0.7732  | 0.7805    |
| 0.0065        | 40.0  | 65360 | 0.9122          | 0.9119 | 0.7977  | 0.8055    |


### Framework versions

- Transformers 4.50.0.dev0
- Pytorch 2.6.0+cu118
- Datasets 3.3.2
- Tokenizers 0.21.0
