---
library_name: transformers
license: mit
base_model: FacebookAI/roberta-large
tags:
- generated_from_trainer
metrics:
- spearmanr
model-index:
- name: roberta-large-num=40
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# roberta-large-num=40

This model is a fine-tuned version of [FacebookAI/roberta-large](https://huggingface.co/FacebookAI/roberta-large) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 0.8514
- Mse: 0.8514
- Pearson: 0.7818
- Spearmanr: 0.7985

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
| 1.0454        | 1.0   | 1634  | 1.0726          | 1.0726 | 0.6745  | 0.6811    |
| 0.6302        | 2.0   | 3268  | 1.1026          | 1.1024 | 0.7094  | 0.7160    |
| 0.447         | 3.0   | 4902  | 0.9070          | 0.9071 | 0.7393  | 0.7482    |
| 0.3038        | 4.0   | 6536  | 0.7734          | 0.7734 | 0.7815  | 0.7908    |
| 0.2069        | 5.0   | 8170  | 0.8119          | 0.8118 | 0.8009  | 0.8070    |
| 0.1375        | 6.0   | 9804  | 0.8022          | 0.8022 | 0.8085  | 0.8179    |
| 0.0998        | 7.0   | 11438 | 0.9700          | 0.9702 | 0.7784  | 0.7857    |
| 0.0739        | 8.0   | 13072 | 1.0172          | 1.0170 | 0.7600  | 0.7731    |
| 0.0621        | 9.0   | 14706 | 0.7508          | 0.7511 | 0.8015  | 0.8133    |
| 0.0522        | 10.0  | 16340 | 0.9481          | 0.9478 | 0.7711  | 0.7816    |
| 0.0428        | 11.0  | 17974 | 0.8033          | 0.8035 | 0.8003  | 0.8076    |
| 0.0359        | 12.0  | 19608 | 0.7930          | 0.7929 | 0.7990  | 0.8081    |
| 0.0354        | 13.0  | 21242 | 0.7877          | 0.7877 | 0.8105  | 0.8187    |
| 0.0327        | 14.0  | 22876 | 0.7981          | 0.7982 | 0.8113  | 0.8222    |
| 0.0287        | 15.0  | 24510 | 0.8191          | 0.8190 | 0.7990  | 0.8118    |
| 0.0268        | 16.0  | 26144 | 0.9833          | 0.9834 | 0.7478  | 0.7625    |
| 0.0231        | 17.0  | 27778 | 0.8239          | 0.8237 | 0.7958  | 0.8084    |
| 0.0202        | 18.0  | 29412 | 0.9154          | 0.9152 | 0.7735  | 0.7796    |
| 0.0203        | 19.0  | 31046 | 0.9315          | 0.9314 | 0.7811  | 0.7943    |
| 0.0184        | 20.0  | 32680 | 0.8719          | 0.8719 | 0.7720  | 0.7838    |
| 0.0176        | 21.0  | 34314 | 0.8849          | 0.8847 | 0.7853  | 0.7969    |
| 0.0155        | 22.0  | 35948 | 0.8631          | 0.8630 | 0.7932  | 0.8087    |
| 0.0146        | 23.0  | 37582 | 0.8744          | 0.8743 | 0.7811  | 0.7942    |
| 0.0134        | 24.0  | 39216 | 0.8717          | 0.8721 | 0.7848  | 0.8002    |
| 0.0134        | 25.0  | 40850 | 0.9405          | 0.9402 | 0.7698  | 0.7823    |
| 0.0139        | 26.0  | 42484 | 1.0385          | 1.0387 | 0.7643  | 0.7742    |
| 0.0115        | 27.0  | 44118 | 0.8943          | 0.8944 | 0.7835  | 0.7980    |
| 0.011         | 28.0  | 45752 | 0.9566          | 0.9566 | 0.7763  | 0.7930    |
| 0.0123        | 29.0  | 47386 | 1.0450          | 1.0448 | 0.7525  | 0.7678    |
| 0.0113        | 30.0  | 49020 | 0.8189          | 0.8187 | 0.7884  | 0.8034    |
| 0.0116        | 31.0  | 50654 | 0.7780          | 0.7780 | 0.8108  | 0.8236    |
| 0.0146        | 32.0  | 52288 | 0.8131          | 0.8132 | 0.7908  | 0.8058    |
| 0.0116        | 33.0  | 53922 | 0.8801          | 0.8802 | 0.7830  | 0.7917    |
| 0.0084        | 34.0  | 55556 | 0.9372          | 0.9372 | 0.7684  | 0.7788    |
| 0.0132        | 35.0  | 57190 | 0.8845          | 0.8844 | 0.7849  | 0.7982    |
| 0.0118        | 36.0  | 58824 | 0.8751          | 0.8750 | 0.7966  | 0.8111    |
| 0.0091        | 37.0  | 60458 | 0.8742          | 0.8740 | 0.7895  | 0.8029    |
| 0.012         | 38.0  | 62092 | 0.8531          | 0.8534 | 0.7773  | 0.7917    |
| 0.0104        | 39.0  | 63726 | 0.9183          | 0.9182 | 0.7757  | 0.7911    |
| 0.0093        | 40.0  | 65360 | 0.8514          | 0.8514 | 0.7818  | 0.7985    |


### Framework versions

- Transformers 4.50.0.dev0
- Pytorch 2.6.0+cu118
- Datasets 3.3.2
- Tokenizers 0.21.0
