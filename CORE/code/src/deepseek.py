from openai import OpenAI
import json 
from tqdm import tqdm

DEEPSEEK_API_KEY = "sk-15df5d1fbd40438b84f468ddbaf4f87e"

class DeepSeekLarge:
    def __init__(self, name): 
        self.name = name
        self.client = OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")
    
    def generate(self, prompts, ids, output_path): 
        count = 0
        for prompt, id in tqdm(zip(prompts, ids)): 
            # retry 3 times if error is found
            try_count = 0
            response = None
            while try_count < 3: 
                try: 
                    response = self.client.chat.completions.create(
                        model="deepseek-reasoner",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant"},
                            {"role": "user", "content": prompt},
                        ],
                        stream=False
                    )
                    break
                except Exception as e: 
                    try_count += 1
                    continue 
            if response: 
                output = response.choices[0].message.content
            else: 
                output = None 

            with open(output_path, "a") as f: 
                json.dump({
                    "id": id,
                    "output": output
                }, f)
                f.write("\n")

            count += 1
