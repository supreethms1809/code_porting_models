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
    default="/Users/ssuresh/aiml/code_porting_models/model_config/DeepSeek-R1-Distill-Qwen-1.5B.yaml",
    #default="/home/sureshm/code_porting_models/model_config/DeepSeek-R1-Distill-Qwen-1.5B.yaml",
    help="Path to the configuration file."
)
args = parser.parse_args()

config_path = args.config_path
if not os.path.exists(config_path):
    raise FileNotFoundError(f"Configuration file not found: {config_path}")
else:
    logging.info(f"Using configuration from: {config_path}")

# Ensure the src directory is in the Python path
src_dir = '/Users/ssuresh/aiml/code_porting_models/'
#src_dir = '/home/sureshm/code_porting_models/'
if src_dir not in sys.path:
    sys.path.append(src_dir)

from src.train_models.evaluate import Evaluate
from src.processdataset.process_babeltower_dataset import ProcessBabelTowerDataset, ProcessBabelTowerTestValDataset
from src.models.qwen_ds.qwen_ds_base import QwenDSBase
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
train = model_arguments.get('train', False)
if not train:
    ds = Dataset.load_from_disk(dataset_path)
    logging.info(
        f"Loaded dataset from disk: {ds}. Number of entries: {len(ds)}."
    )
    btpre = ProcessBabelTowerTestValDataset(
        dataset=ds
    )
    pds = btpre.process_dataset_for_eval(ds=ds)
    logging.info(f"Processed dataset for evaluation: {pds} entries.")


model = QwenDSBase(model_arguments)
model.load_model_and_tokenizer()
query = model.structure_message(pds["final_query"][0])
logging.info(f"Structured message: {query}")
model_output = model.generate_code(query)
logging.info(f"Model output: {model_output}")