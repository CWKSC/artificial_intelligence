from transformers import AutoTokenizer, GPTQConfig, AutoModelForCausalLM

import torch
print(torch.cuda.is_available())

model_id = "facebook/opt-125m"
tokenizer = AutoTokenizer.from_pretrained(model_id)
gptq_config = GPTQConfig(bits=4, dataset = "c4", tokenizer=tokenizer)

model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", quantization_config=gptq_config)



