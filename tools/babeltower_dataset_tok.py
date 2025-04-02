import json
import os
import logging
from datasets import Dataset

logging.basicConfig(level=logging.INFO)
# data_list = ['cpp.mono.train.tok', 'cuda.mono.train.tok']

# def parse_line(line):
#     """
#     Parse a line into an identifier and code snippet.
#     Expected line format: <identifier> | <code snippet>
#     """
#     parts = line.split('|', 1)
#     if len(parts) != 2:
#         logging.error(f"Line does not conform to expected format: {line.strip()}")
#         return None, None
#     identifier = parts[0].strip()
#     code_snippet = parts[1].strip()
#     return identifier, code_snippet

# def read_and_parse_file(file_path):
#     results = []
#     try:
#         for file_name in file_path:
#             with open(file_name, 'r') as f:
#                 for line in f:
#                     if not line.strip():
#                         continue  # skip empty lines
#                     identifier, code = parse_line(line)
#                     if identifier is not None and code is not None:
#                         results.append((identifier, code))
#         logging.info(f"Successfully parsed file: {file_name}")
#     except Exception as e:
#         logging.error(f"Error reading file {file_name}: {e}")
#     return results

# file_path = [ os.path.join(os.environ.get('BASE_DIR', '/Users/ssuresh/aiml/code_porting_models/'), 'dataset', data_list[0]),
#                 os.path.join(os.environ.get('BASE_DIR', '/Users/ssuresh/aiml/code_porting_models/'), 'dataset', data_list[1])
#                 ]

# parsed_lines = read_and_parse_file(file_path)
# # Build lists of identifiers and codes
# identifiers = [identifier for identifier, code in parsed_lines]
# codes = [code for identifier, code in parsed_lines]

# # Create a Hugging Face Dataset from list of values
# ds = Dataset.from_dict({"identifier": identifiers, "code": codes})

# # Save the dataset to a JSON file
# output_dir = os.path.join(os.environ.get('BASE_DIR', '/Users/ssuresh/aiml/code_porting_models/dataset'), 'babeltower')
# ds.save_to_disk(output_dir)
# logging.info(f"hf_dataset identifier: {ds}")

# load the dataset from disk
output_dir = os.path.join(os.environ.get('BASE_DIR', '/Users/ssuresh/aiml/code_porting_models/dataset'), 'babeltower')
ds = Dataset.load_from_disk(output_dir)
logging.info(f"hf_dataset identifier: {ds}")