from datasets import load_dataset
import os
import yaml
import logging
# Set up logging
logging.basicConfig(level=logging.INFO)

# Define the dataset directory
dataset_dir = os.path.join(os.environ.get('BASE_DIR', '/workspace/code_porting_models'), 'dataset')

# Load the dataset and specify the cache directory
ds = load_dataset("OpenCoder-LLM/RefineCode-code-corpus-meta", cache_dir=dataset_dir)

logging.info("Dataset loaded successfully.")
logging.info(f"Dataset: {ds}")
logging.info(f"Dataset keys: {ds.keys()}")
unique_keys = set(ds['The_Stack_V2']['program_lang'])
logging.info(f"Unique keys: {unique_keys}")
