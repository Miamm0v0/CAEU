import json 
from tqdm import tqdm
import os

import transformers
from transformers import AutoProcessor
from transformers import AutoTokenizer, AutoModelForCausalLM
import requests
import torch

class Phi: 
    def __init__(self, name): 
        self.name = name
        self.model_id = "microsoft/phi-4"
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_id,
            model_kwargs={"torch_dtype": "auto"},
            device_map="auto",
        )
    
    def generate(self, prompts, ids, output_path): 
        count = 0
        system_prompt = "You are a helpful model, capable of thinking like a human being. You will respond following the exact instructions provided in the given prompt."
        for prompt, id in tqdm(zip(prompts, ids)):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]

            outputs = self.pipeline(messages, max_new_tokens=256)
            output = outputs[0]["generated_text"][-1]
            
            with open(output_path, "a") as f: 
                json.dump({
                    "id": id,
                    "output": output
                }, f)
                f.write("\n")
            count += 1

class PhiReasoning: 
    def __init__(self, name): 
        self.name = name
        self.model_id = "microsoft/phi-4-reasoning"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, device_map="auto").eval()
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=self.model_id,
            model_kwargs={"torch_dtype": "auto"},
            device_map="auto",
        )
    
    
    def generate(self, prompts, ids, output_path): 
        count = 0
        system_prompt = "You are a helpful model, capable of thinking like a human being. You will respond following the exact instructions provided in the given prompt."
        for prompt, id in tqdm(zip(prompts, ids)):
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            outputs = self.pipeline(messages, max_new_tokens=1024)
            output = outputs[0]["generated_text"][-1]
            with open(output_path, "a") as f: 
                json.dump({
                    "id": id,
                    "output": output
                }, f)
                f.write("\n")
            count += 1

