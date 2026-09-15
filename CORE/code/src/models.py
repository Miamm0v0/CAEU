import os

os.environ['HF_HOME'] = "/work/hdd/bbmr/sbhattacharyya1/hub_home"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, Gemma3ForConditionalGeneration, MllamaForConditionalGeneration
from huggingface_hub import login
import torch
from qwen_vl_utils import process_vision_info
from vllm import LLM, SamplingParams
import PIL.Image
import PIL
import pandas as pd
import json
import argparse
from tqdm import tqdm

class ModelClass: 
    def __init__(self, name):
        self.name = name
        self.model, self.sampling_params = self.model_setup()
    
    def model_setup(self): 
        if self.name == "qwen":
            model_id = "Qwen/Qwen3-32B"
        if self.name == "qwen-reasoner": 
            model_id = "Qwen/QwQ-32B"
        # elif self.name == "deepseek": 
            # model_id = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"
        elif self.name == "llama": 
            model_id = "meta-llama/Meta-Llama-3-8B-Instruct"
        model = LLM(
                model=model_id,
                max_model_len=2048,
                trust_remote_code=True,
                # gpu_memory_utilization=0.70,
                tensor_parallel_size=2
            )
        sampling_params = SamplingParams(
                temperature=0.5,
                max_tokens=1000
            )
        return model, sampling_params 
    
    def generate(self, prompts):
        outputs = self.model.generate(prompts, self.sampling_params, use_tqdm=True)
        return outputs