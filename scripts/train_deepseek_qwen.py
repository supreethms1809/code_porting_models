import os
import sys
import yaml
# Add the root directory of the project to the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.logging import setup_log

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

save_directory = f"/home/ssuresh/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
tokenizer.save_pretrained(save_directory)
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
model.save_pretrained(save_directory)

logger = setup_log()
logger.info("Model and tokenizer loaded successfully.")

def generate_code(prompt):
    inputs = tokenizer(prompt, return_tensors="pt")
    logger.info(f"Inputs: {model.generate.__dict__}")
    outputs = model.generate(**inputs)
    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_code


code_prompt = '''
write a c++ code to find the sum of two arrays
'''
generated_code = generate_code(code_prompt)
print("Generated Code:")
print(generated_code)

