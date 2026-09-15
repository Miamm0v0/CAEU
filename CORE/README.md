This file provides all required information to run the code. However, we would advise against running the entire suite of evaluation, as for the proprietary models (GPT-o4 mini, DeepSeek R1, and Gemini 2.5 Flash), it would require setting up an API/credit account. For the open-source models (Phi, Qwen and LLama), we used A40 and A100 GPUs. Qwen QwQ requires at least 2 of such GPUs to load and run. Given that LLMs are stochastic next-token predictors, re-sampling the data from the LLMs may or may not lead to the same responses, even when all other parameters are kept the same. We thus make all of the generated data (and the used prompts) directly available to enable the running of the resulting analysis.

- #### System Requirements: 
    - All system requirements are provided under requirements.txt.
    - To run the full suite of LLM evaluations, the following things would be required: 
        - billing account set up on OpenAI and API key created for querying GPT models: https://auth.openai.com/log-in 
        - billing account set up on the Google Developer Console, and API key created for Gemini models: https://aistudio.google.com/
        - biling account set up on DeepSeek platform, and API key created for DeepSeek R1: https://platform.deepseek.com/. Note that DeepSeek R1 is now replaced by a newer model, and it may not be possible to directly query it anymore.
    - To run the open-source models, access to 2 A40 (GPU memory (GB) = 48 DDR6 with ECC) or A100 GPUs (GPU memory (GB) = 40). 

- #### Installation Guide: 
    - Step 1: Ensure Python is installed. 
    - Step 2: Ensure Anaconda is installed. Detailed information on it can be found here: https://www.anaconda.com/download. On most distributed servers or HPC clusters, a preloaded version of anaconda should be available to use.
    - Step 3: Create a new conda environment using: `conda create --name <name_of_your_environment>`
    - Step 4: Install pip. This can be done using `conda install pip`
    - Step 5: Install all of the packages listed under requirements.txt. This can be done by `pip install -r requirements.txt`
    Typically, the entire installation process should take at most 1 hour to complete.

- #### Demo and Instructions for Use:
    - To re-run LLM data generation: 
        - The following code is required to be run (ideally submitted as a batch job) from within the src directory: 
            `python3 -u evaluate.py -p <prompt_path> -m <name_of_model> -o <output_path>`
        Running the data generation for the vanilla set of prompts (N=4658) can take upto 48 hours for DeepSeek R1, ~24 hours for Gemini, GPT and Phi, and ~2 hours for Qwen and LLama.
    
    - To re-run each of the other sets of analyses, the code in the provided jupyter notebooks are required to be run. For this, ensure that jupyter is installed (`pip install jupyter`).
    - All of the code automatically uses the data from the benchmark and/or the data generated through the LLM evaluation process. In some cases, the only requirement would be to change the path provided in code to the relative path, instead of the absolute path provided. 
