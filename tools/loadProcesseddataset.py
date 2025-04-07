from datasets import load_dataset, load_from_disk, Dataset
import os
import yaml
import logging
import sys
# Set up logging
logging.basicConfig(level=logging.INFO)
# Ensure the src directory is in the Python path
src_dir = '/home/sureshm/code_porting_models/'
if src_dir not in sys.path:
    sys.path.append(src_dir)

process = True

if process:
    from src.processdataset.process_babeltower_dataset import process_dataset_hf_format

    # Define the dataset directory
    dataset_dir = os.path.join(os.environ.get('BASE_DIR', '/home/sureshm/code_porting_models'), 'dataset')

    ds = process_dataset_hf_format( 
        cpp_path_test=os.path.join(dataset_dir, 'cpp2cuda', 'cpp.para.test.tok'),
        cuda_path_test=os.path.join(dataset_dir, 'cpp2cuda', 'cuda.para.test.tok'),
        cpp_path_val=os.path.join(dataset_dir, 'cpp2cuda', 'cpp.para.valid.tok'),
        cuda_path_val=os.path.join(dataset_dir, 'cpp2cuda', 'cuda.para.valid.tok'), 
        save_dir=os.path.join(dataset_dir, 'babeltower_test_val')
    )
    logging.info(
        f"Processed dataset saved to: {os.path.join(dataset_dir, 'babeltower_test_val')}. \n"
        "dataset is {ds}"
    )




