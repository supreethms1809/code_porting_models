from datasets import load_dataset, load_from_disk
import os
import yaml
import logging
# Set up logging
logging.basicConfig(level=logging.INFO)

# Define the dataset directory
dataset_dir = os.path.join(os.environ.get('BASE_DIR', '/home/sureshm/code_porting_models'), 'dataset')

dataset_path = os.path.join(dataset_dir, "code_dataset/V2combined_dataset")
ds_local = load_from_disk(dataset_path)
logging.info(f"Dataset: {ds_local}")


