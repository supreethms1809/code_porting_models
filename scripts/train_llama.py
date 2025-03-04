import os
import sys
import yaml

# Add the root directory of the project to the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.utils.logging import setup_log
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("meta-llama/CodeLlama-7b-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/CodeLlama-7b-hf")

logger = setup_log()
logger.info("Model and tokenizer loaded successfully.")

def generate_code(prompt):
    """
    Generate code using the loaded model and tokenizer.
    
    Args:
        prompt (str): The input prompt for code generation.
        
    Returns:
        str: The generated code.
    """
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(**inputs)
    generated_code = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return generated_code


code_prompt = "def add(a, b):\n    # This function adds two numbers\n    return a + b"
generated_code = generate_code(code_prompt)
print("Generated Code:")
print(generated_code)