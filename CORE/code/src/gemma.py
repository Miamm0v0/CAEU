import json 
from tqdm import tqdm
import os

from transformers import AutoProcessor, Gemma3ForConditionalGeneration
import requests
import torch

class Gemma: 
    def __init__(self, name): 
        self.name = name
        self.model_id = "google/gemma-3-27b-pt"
        self.model = Gemma3ForConditionalGeneration.from_pretrained(self.model_id,
                            device_map="auto").eval()
        self.processor = AutoProcessor.from_pretrained(self.model_id)

    
    def generate(self, prompts, ids, output_path): 
        count = 0
        for prompt, id in tqdm(zip(prompts, ids)):
            model_inputs = self.processor(text=prompt, images=None, return_tensors='pt')
            model_inputs = model_inputs.to(self.model.device)
            input_len = model_inputs["input_ids"].shape[-1]

            with torch.inference_mode():
                generation = self.model.generate(**model_inputs, max_new_tokens=256, do_sample=False)
                generation = generation[0][input_len:]
            
            with open(output_path, "a") as f: 
                json.dump({
                    "id": id,
                    "output": generation
                }, f)
                f.write("\n")
            count += 1
