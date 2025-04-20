import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import os


tokenizer = AutoTokenizer.from_pretrained('app/google/gemma-2-2b-it')
model = AutoModelForCausalLM.from_pretrained('app/google/gemma-2-2b-it')

input_text = "이순신장군이 누구입니까?"
input_ids = tokenizer(input_text, return_tensors="pt")

outputs = model.generate(**input_ids, max_length=512)
print(tokenizer.decode(outputs[0]))
