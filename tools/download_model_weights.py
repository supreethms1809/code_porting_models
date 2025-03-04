from transformers import AutoTokenizer, AutoModelForCausalLM

# Specify the model name from the Hugging Face Hub
model_name = ["deepseek-ai/DeepSeek-R1-Distill-Llama-8B", 
              "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
              "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",]

for name in model_name:
    print(f"Downloading model {name} to /workspace/models/{name}")
    
    # Define the local directory to save the model and tokenizer
    save_directory = f"/workspace/models/{name}"
    
    # Download and save the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.save_pretrained(save_directory)

    # Download and save the model
    model = AutoModelForCausalLM.from_pretrained(name)
    model.save_pretrained(save_directory)
    print(f"Model {name} downloaded and saved to {save_directory}")

# import os
# import sys
# import yaml

# # Add the root directory of the project to the PYTHONPATH
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from src.utils.logging import setup_log
# from transformers import AutoTokenizer, AutoModelForCausalLM

# #save_directory = f"/home/ssuresh/models/meta-llama/CodeLlama-7b-hf"

# # tokenizer = AutoTokenizer.from_pretrained(model_path)
# # model = AutoModelForCausalLM.from_pretrained(model_path)
# tokenizer = AutoTokenizer.from_pretrained("meta-llama/CodeLlama-7b-hf")
# #tokenizer.save_pretrained(save_directory)
# model = AutoModelForCausalLM.from_pretrained("meta-llama/CodeLlama-7b-hf")
# #model.save_pretrained(save_directory)

# # logger = setup_log()
# # logger.info("Model and tokenizer loaded successfully.")
# # logger.info(f"Model: {model}")