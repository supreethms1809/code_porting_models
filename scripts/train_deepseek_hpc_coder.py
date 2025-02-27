import os
import sys
import yaml
# Add the root directory of the project to the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.logging import setup_log

from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained("hpcgroup/hpc-coder-v2-16b", trust_remote_code=True)
