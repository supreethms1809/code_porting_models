import torch
import logging
import os
import yaml
import sys
import argparse
logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser(description="Collect metrics for the BabelTower dataset.")
parser.add_argument(
    "--config_path",
    type=str,
    default="/home/sureshm/code_porting_models/model_config/DeepSeek-R1-Distill-Qwen-1.5B.yaml",
    help="Path to the configuration file."
)
args = parser.parse_args()

config_path = args.config_path
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Configuration file not found: {config_path}")
else:
    logging.info(f"Using configuration from: {config_path}")

# Ensure the src directory is in the Python path
src_dir = '/home/sureshm/code_porting_models/'
if src_dir not in sys.path:
    sys.path.append(src_dir)

from src.train_models.evaluate import Evaluate
from src.processdataset.process_babeltower_dataset import ProcessBabelTowerDataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from datasets import load_dataset, load_from_disk, Dataset

model_arguments = yaml.safe_load(open(config_path, "r"))

# Load the tokenizer and model
model = AutoModelForCausalLM.from_pretrained(model_arguments.get('hf_hub_repo_id'))
tokenizer = AutoTokenizer.from_pretrained(model_arguments.get('hf_hub_repo_id'))

if model is None or tokenizer is None:
    raise RuntimeError(
        f"Failed to load the model or tokenizer from the specified path: {model_arguments.hf_hub_repo_id}"
    )

config = model.config
dataset_path = model_arguments.get('dataset_path')
dataset_hf = model_arguments.get('hf_dataset_repo_id')

dataset = Dataset.load_from_disk(dataset_path)
logging.info(
    f"Loaded dataset from disk: {dataset_path}. Number of entries: {len(dataset)}."
)
btpre = ProcessBabelTowerDataset(
    dataset=dataset
)
pDataset = btpre.process(task="val")
logging.info(
    f"Processed dataset for evaluation: {pDataset} entries."
)
