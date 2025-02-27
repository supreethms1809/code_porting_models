import os
import sys
import yaml

# Add the root directory of the project to the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.logging import setup_log
from transformers import AutoTokenizer, AutoModelForCausalLM

save_directory = f"/home/ssuresh/models/meta-llama/CodeLlama-7b-hf"

# tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained("meta-llama/CodeLlama-7b-hf")
tokenizer.save_pretrained(save_directory)
model = AutoModelForCausalLM.from_pretrained("meta-llama/CodeLlama-7b-hf")
model.save_pretrained(save_directory)

# logger = setup_log()
# logger.info("Model and tokenizer loaded successfully.")
# logger.info(f"Model: {model}")