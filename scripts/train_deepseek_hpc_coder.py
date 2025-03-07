import os
import sys
import yaml
# Add the root directory of the project to the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.logging import setup_log

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer

save_directory = f"/home/ssuresh/models/hpcgroup/hpc-coder-v2-16b"

tokenizer = AutoTokenizer.from_pretrained("hpcgroup/hpc-coder-v2-16b", trust_remote_code=True)
tokenizer.save_pretrained("hpc-coder-v2-16b")
model = AutoModelForCausalLM.from_pretrained("hpcgroup/hpc-coder-v2-16b", trust_remote_code=True)
model.save_pretrained("hpc-coder-v2-16b")

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

