'''
train_gpt.py
This script is used to train a GPT model using the Hugging Face Transformers library.
It loads a dataset, tokenizes it, and trains the model using the specified training arguments.
It also saves the trained model and tokenizer to a specified directory.
Configuration: model_config/gpt.yaml
Usage:
    python train_gpt.py 

     
'''

import os
import yaml
import argparse
import logging
import torch

# Set up logging
logging.basicConfig(level=logging.INFO)

# Check for available device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

logging.info(f"Using device: {device}")


def main():
    logging.info("Loading config...")

    # Get the current working directory
    base_dir = os.environ.get('BASE_DIR', '/workspace/code_porting_models')
    logging.info(f"Base directory: {base_dir}")
    
    # Load configuration
    config_path = os.path.join(base_dir, "model_config/gpt.yaml")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)

    logging.info("Configuration loaded successfully.")
    logging.info(f"Configuration: {config}")


if __name__ == "__main__":
    main()