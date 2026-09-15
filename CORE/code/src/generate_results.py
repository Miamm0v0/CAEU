import os

os.environ['HF_HOME'] = "/work/hdd/bbmr/sbhattacharyya1/hub_home"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
import pandas as pd
import json
import argparse
from tqdm import tqdm

from deepseek import DeepSeekLarge
from gemini import Gemini
from gpt import GPT
from gemma import Gemma
from phi import Phi, PhiReasoning
from mistral import Mistral

def get_args(): 
    parser = argparse.ArgumentParser(description="Enter custom options for prompt set, and model.")
    parser.add_argument("-p", "--prompt_path", required=True, type=str)
    parser.add_argument("-m", "--model_name", required=True, type=str)
    parser.add_argument("-o", "--output_path", required=True, type=str)

    args = parser.parse_args()
    return args.prompt_path, args.model_name, args.output_path

if __name__ == "__main__":
    print(f"Hit start of the code!")
    prompt_path, model_name, output_path = get_args()
    prompt_list = []
    id_list = []

    processed_ids = set()
    if os.path.exists(output_path): 
        with open(output_path, "r") as f: 
            for line in f.readlines(): 
                curr_json = json.loads(line)
                id = curr_json["id"]
                processed_ids.add(id)
    print(f"Number of existing results = {len(processed_ids)}")
    
    with open(prompt_path, "r") as f: 
        for line in f.readlines(): 
            curr_json = json.loads(line)
            if curr_json["id"] in processed_ids: 
                continue
            id_list.append(curr_json["id"])
            prompt_list.append(curr_json["prompt"])
    print(f"Length of prompts = {len(prompt_list)}")

    if model_name == "deepseek":  # only support one deepseek model now
        model = DeepSeekLarge(model_name)
        model.generate(prompt_list, id_list, output_path)
    elif model_name == "gpt": 
        model = GPT(model_name)
        model.generate(prompt_list, id_list, output_path)
    elif model_name == "gemini":
        model = Gemini(model_name)
        model.generate(prompt_list, id_list, output_path)
    elif model_name == "gemma": 
        model = Gemma(model_name)
        model.generate(prompt_list, id_list, output_path)
    elif model_name == "phi":
        model = Phi(model_name)
        model.generate(prompt_list, id_list, output_path)
    elif model_name == "phi-reasoner":
        model = PhiReasoning(model_name)
        model.generate(prompt_list, id_list, output_path)
    elif model_name == "mistral": 
        model = Mistral(model_name)
        model.generate(prompt_list, id_list, output_path)
    else:
        from models import ModelClass
        
        generator_model = ModelClass(model_name)
        batch_size = 2048 
        n = len(prompt_list)
        count = 0 
        for i in range(0, n, batch_size): 
            print(f"Processing Batch {count+1}")
            end = min(i+batch_size, n)

            curr_ids = id_list[i:end]
            curr_prompts = prompt_list[i:end]
            print(f"Generating Outputs:")
            outputs = generator_model.generate(curr_prompts)

            for output, id in zip(outputs, curr_ids): 
                with open(output_path, "a") as f: 
                    json.dump({
                        "id": id,
                        "output": output.outputs[0].text
                    }, f)
                    f.write("\n")
            count += 1
        
                    



        
        
