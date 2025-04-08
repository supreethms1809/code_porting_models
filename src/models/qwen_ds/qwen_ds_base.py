import os
import yaml
import logging
import sys
from datasets import Dataset
import argparse
from src.processdataset.process_babeltower_dataset import ProcessBabelTowerDataset
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from transformers import pipeline
from src.train_models.evaluate import Evaluate

# Set up logging
logging.basicConfig(level=logging.INFO)
# Ensure the src directory is in the Python path
#src_dir = '/Users/ssuresh/aiml/code_porting_models/'
src_dir = '/home/sureshm/code_porting_models/'
if src_dir not in sys.path:
    sys.path.append(src_dir)

parser = argparse.ArgumentParser(description="Load and process the BabelTower dataset.")
parser.add_argument(
    "--config_path",
    type=str,
    #default="/Users/ssuresh/aiml/code_porting_models/model_config/DeepSeek-R1-Distill-Qwen-1.5B.yaml",
    default="/home/sureshm/code_porting_models/model_config/DeepSeek-R1-Distill-Qwen-1.5B.yaml",
    help="Path to the configuration file."
)
args = parser.parse_args()
config_path = args.config_path


class QwenDSBase:
    def __init__(self, model_arguments):
        self.model_arguments = model_arguments
        self.model = None
        self.tokenizer = None
        self.dataset = None
        self.eval_pipeline = None

    def load_model_and_tokenizer(self):
        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_arguments.get('hf_hub_repo_id'))
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_arguments.get('hf_hub_repo_id'))
            if self.model is None or self.tokenizer is None:
                raise RuntimeError(
                    f"Failed to load the model or tokenizer from the specified path: {self.model_arguments.hf_hub_repo_id}"
                )
            logging.info(f"Model and tokenizer loaded successfully.")
            
        except Exception as e:
            logging.error(f"Error loading model and tokenizer: {e}")
            raise e
        
        self.eval_pipeline = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            truncation=False,
            max_length=self.model.config.max_position_embeddings
        )
        return self.model, self.tokenizer, self.eval_pipeline

    def structure_message(self, message):
        try:
            structured_message = [{
                "role": "user",
                "content": message,
            }]
            return structured_message
        except Exception as e:
            logging.error(f"Error structuring message: {e}")
            raise e

    def generate_code(self, input_text):
        try:
            if self.eval_pipeline is None:
                raise RuntimeError("Evaluation pipeline is not initialized.")
            generated_code = self.eval_pipeline(input_text)
            return generated_code
        except Exception as e:
            logging.error(f"Error generating code: {e}")
            raise e
        return generated_code
