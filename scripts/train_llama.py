import os
import sys
import yaml

# Add the root directory of the project to the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.logging import setup_log
from transformers import AutoTokenizer, AutoModelForCausalLM

save_directory = f"/home/ssuresh/models/meta-llama/CodeLlama-7b-Instruct-hf"

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/CodeLlama-7b-Instruct-hf")
tokenizer.save_pretrained(save_directory)
model = AutoModelForCausalLM.from_pretrained("meta-llama/CodeLlama-7b-Instruct-hf")
model.save_pretrained(save_directory)

logger = setup_log()
logger.info("Model and tokenizer loaded successfully.")

def generate_code(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs)
    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_code


code_prompt = '''
write a c++ code to find the sum of two arrays
'''
generated_code = generate_code(code_prompt)
print("Generated Code:")
print(generated_code)