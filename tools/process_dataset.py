from datasets import load_dataset, load_from_disk, concatenate_datasets
import os
import yaml
import logging
# Set up logging
logging.basicConfig(level=logging.INFO)

# Define the dataset directory
dataset_dir = os.path.join(os.environ.get('BASE_DIR', '/home/sureshm/code_porting_models'), 'dataset')

# Load the dataset and specify the cache directory
#ds = load_dataset("OpenCoder-LLM/RefineCode-code-corpus-meta", cache_dir=dataset_dir)

# logging.info("Dataset loaded successfully.")
# logging.info(f"Dataset: {ds}")
# logging.info(f"Dataset keys: {ds.keys()}")
# #unique_keys = set(ds['The_Stack_V2']['program_lang'])
# #logging.info(f"Unique keys: {unique_keys}")

# filtered_ds = ds['The_Stack_V2'].filter(lambda ds: ds['program_lang'] in ['unified parallel c','objective-c++','f*','f#','fortran free form','cpp','fortran','ooc','cuda','c','opencl'], num_proc=72)
# logging.info(f"Filtered dataset: {filtered_ds}")
# save_path = os.path.join(dataset_dir, "filteredOpenCoder_llm_refine_code_corpus_dataset")
# filtered_ds.save_to_disk(save_path)

# c++
# c
# unified-parallel-c
# opencl
# cuda
# fortran
# metal

######## bigcode the stack dataset
# ds = load_dataset("bigcode/the-stack", data_dir="data/metal",  cache_dir=dataset_dir)
# logging.info(f"Dataset: {ds}")
# ds.save_to_disk(os.path.join(dataset_dir, "code_dataset/metal"))

# # Define the dataset directory
# dataset_dir = os.path.join(os.environ.get('BASE_DIR', '/home/sureshm/code_porting_models'), 'dataset')

# dataset_path = os.path.join(dataset_dir, "bigcode___the-stack")
# ds = load_from_disk(dataset_path)
# logging.info(f"Dataset: {ds}")

from datasets import load_from_disk, concatenate_datasets
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define the base dataset directory
dataset_dir = os.path.join(os.environ.get('BASE_DIR', '/home/sureshm/code_porting_models'), 'dataset')
dataset_dir = os.path.join(dataset_dir, "code_dataset")
# List to hold all loaded datasets
# Iterate over each subdirectory in the dataset directory
all_datasets = []

for subdir in os.listdir(dataset_dir):
    subdir_path = os.path.join(dataset_dir, subdir)
    if os.path.isdir(subdir_path):
        try:
            if subdir == "c" or subdir == "c++" or subdir == "cuda" or subdir == "opencl":
                dataset_dict = load_from_disk(subdir_path)
                # Extract the 'train' split from each dataset
                if 'train' in dataset_dict:
                    ds = dataset_dict['train']
                    all_datasets.append(ds)
                    logging.info(f"Loaded 'train' split from {subdir_path} with {ds.num_rows} rows")
                else:
                    logging.warning(f"No 'train' split found in {subdir_path}")
        except Exception as e:
            logging.warning(f"Skipping {subdir_path}: {e}")

# Concatenate all loaded datasets into one if at least one was successfully loaded
if all_datasets:
    v2combined_dataset = concatenate_datasets(all_datasets)
    save_path = os.path.join(dataset_dir, "V2combined_dataset")
    v2combined_dataset.save_to_disk(save_path)
    logging.info(f"Combined dataset created with {v2combined_dataset.num_rows} total rows")
else:
    logging.error("No valid datasets found to combine.")