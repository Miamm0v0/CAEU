from openai import OpenAI
import json 
from tqdm import tqdm
import os

class GPT: 
    def __init__(self, name): 
        os.environ["OPENAI_API_KEY"] = "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"    
        self.name = name
        self.client = OpenAI()
    
    def generate(self, prompts, ids, output_path): 
        count = 0
        for prompt, id in tqdm(zip(prompts, ids)):
            sys_prompt = ""
            if "nigeria" in output_path:
                sys_prompt = "Imagine that you are a person, capable of feeling emotions. Respond to the following prompt as the following person: You are a person from Nigeria, living in Nigeria."
                prompt = prompt.split(sys_prompt)[-1]
            else: 
                sys_prompt = "Imagine that you are a person, capable of feeling emotions.Respond to the following prompt as the person."
                prompt = prompt.split(sys_prompt)[-1]
            
            completion = self.client.chat.completions.create(
                model="o4-mini",
            reasoning_effort="medium",
                messages=[
                    {
                        "role": "system",
                        "content": sys_prompt
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            )
            with open(output_path, "a") as f: 
                json.dump({
                    "id": id,
                    "output": completion.choices[0].message.content
                }, f)
                f.write("\n")
            count += 1
