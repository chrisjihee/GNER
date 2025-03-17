model_specs_2gpu = [
    {"ds_config": "configs/deepspeed/ds0_t5.json", "run_prefix": "FlanT5-Large", "pretrained": "google/flan-t5-large", "train_batch": 16, "train_epochs": 12},
    {"ds_config": "configs/deepspeed/ds0_t5.json", "run_prefix": "FlanT5-Base", "pretrained": "google/flan-t5-base", "train_batch": 32, "train_epochs": 12},
]

model_specs_4gpu_a = [
    {"ds_config": "configs/deepspeed/ds2_llama.json", "run_prefix": "Llama3-1B", "pretrained": "meta-llama/Llama-3.2-1B", "train_batch": 2, "train_epochs": 10},
    {"ds_config": "configs/deepspeed/ds2_llama.json", "run_prefix": "Llama3-3B", "pretrained": "meta-llama/Llama-3.2-3B", "train_batch": 2, "train_epochs": 10},
]

model_specs_4gpu_b = [
    {"ds_config": "configs/deepspeed/ds2_t5.json", "run_prefix": "FlanT5-3B", "pretrained": "google/flan-t5-xl", "train_batch": 8, "train_epochs": 10},
    {"ds_config": "configs/deepspeed/ds2_llama.json", "run_prefix": "Phi3-4B", "pretrained": "microsoft/Phi-3-mini-4k-instruct", "train_batch": 8, "train_epochs": 10},  # modeling_phi3.py: get_max_length -> get_max_cache_shape
]

model_specs_4gpu_c = [
    {"ds_config": "configs/deepspeed/ds2_llama.json", "run_prefix": "Qwen2-1B", "pretrained": "Qwen/Qwen2.5-1.5B", "train_batch": 16, "train_epochs": 10},
    {"ds_config": "configs/deepspeed/ds2_llama.json", "run_prefix": "Qwen2-3B", "pretrained": "Qwen/Qwen2.5-3B", "train_batch": 8, "train_epochs": 10},
]

model_specs_8gpu = [
    # {"ds_config": "configs/deepspeed/ds2_llama.json", "run_prefix": "Phi3-7B", "pretrained": "microsoft/Phi-3-small-8k-instruct", "train_batch": 1, "train_epochs": 6},
    # {"ds_config": "configs/deepspeed/ds2_llama.json", "run_prefix": "Qwen2-7B", "pretrained": "Qwen/Qwen2.5-7B", "train_batch": 1, "train_epochs": 6},
    # {"ds_config": "configs/deepspeed/ds2_llama.json", "run_prefix": "Llama2-7B", "pretrained": "meta-llama/Llama-2-7b-hf", "train_batch": 1, "train_epochs": 6},
    # {"ds_config": "configs/deepspeed/ds2_llama.json", "run_prefix": "Llama3-8B", "pretrained": "meta-llama/Llama-3.1-8B", "train_batch": 1, "train_epochs": 6},
    # {"ds_config": "configs/deepspeed/ds2_llama.json", "run_prefix": "Mistral-7B", "pretrained": "mistralai/Mistral-7B-Instruct-v0.3", "train_batch": 1, "train_epochs": 6},
    # {"ds_config": "configs/deepspeed/ds2_llama.json", "run_prefix": "Ministral-8B", "pretrained": "mistralai/Ministral-8B-Instruct-2410", "train_batch": 1, "train_epochs": 6},
]

model_specs_16gpu = [
    # {"ds_config": "configs/deepspeed/ds3_t5.json", "run_prefix": "FlanT5-11B", "pretrained": "google/flan-t5-xxl", "train_batch": 1, "train_epochs": 6},
    # {"ds_config": "configs/deepspeed/ds3_llama.json", "run_prefix": "Mistral-12B", "pretrained": "mistralai/Mistral-Nemo-Instruct-2407", "train_batch": 1, "train_epochs": 6},
    # {"ds_config": "configs/deepspeed/ds3_llama.json", "run_prefix": "Llama2-13B", "pretrained": "meta-llama/Llama-2-13b-hf", "train_batch": 1, "train_epochs": 6},
    # {"ds_config": "configs/deepspeed/ds3_llama.json", "run_prefix": "Phi4-14B", "pretrained": "microsoft/phi-4", "train_batch": 1, "train_epochs": 6},
    # {"ds_config": "configs/deepspeed/ds3_llama.json", "run_prefix": "Qwen2-14B", "pretrained": "Qwen/Qwen2.5-14B", "train_batch": 1, "train_epochs": 6},
]
