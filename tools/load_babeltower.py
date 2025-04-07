import json
import os
import logging
import logging

logging.basicConfig(level=logging.INFO)

def read_json_file(file_path):
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return None
    results = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                if line.strip():  # skip empty lines
                    results.append(json.loads(line))
        logging.info(f"Successfully read JSON objects from file: {file_path}")
        return results
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from file {file_path}: {e}")
        return None
 
data_list = ['cuda.mono.train.aer.jsonl', 'cpp.mono.train.aer.jsonl', 'cpp.cuda.train.synthetic.jsonl']
ds = {}
data_dir = os.path.join(os.environ.get('BASE_DIR', '/Users/ssuresh/aiml/code_porting_models/'), 'dataset')
for data_file in data_list:
    ds[data_file] = read_json_file(os.path.join(data_dir, data_file))
logging.info(f"File: {ds['cuda.mono.train.aer.jsonl'][0]['tokens']}")
