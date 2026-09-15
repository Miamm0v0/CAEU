from google import genai
import json 
from tqdm import tqdm

# GEMINI_API_KEY = "AIzaSyDg_Dt4UGDh-wccXPt_bPVoOYKWglkpYag"
# GEMINI_API_KEY = "AIzaSyBHLzwViQow-RV_tVRwdFsReYVkfUcElmY"
# GEMINI_API_KEY = "AIzaSyBhuuslF6cypoplArrlDNsWNtPWWVHxwRA"
GEMINI_API_KEY = "AIzaSyCUQYnO8IncICtGFXVMQ8d8eUgZeXBXxDo"

class Gemini:
    def __init__(self, name): 
        self.name = name
        self.client = genai.Client(api_key=GEMINI_API_KEY)
    
    def generate(self, prompts, ids, output_path): 
        count = 0
        for prompt, id in tqdm(zip(prompts, ids)): 
            response = ""
            for i in range(10): 
                try:
                    response = self.client.models.generate_content(
                        model="gemini-2.5-flash", contents=prompt
                    )
                    break
                except Exception as e:
                    print(f"Failed on try {i+1} with error = {e}")
                    continue 
            response_to_write = ""
            if response: 
                response_to_write = response.text
            with open(output_path, "a") as f: 
                json.dump({
                    "id": id,
                    "output": response_to_write
                }, f)
                f.write("\n")

            count += 1
