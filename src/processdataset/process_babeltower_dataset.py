# class to process the BabelTower dataset
from datasets import Dataset
class ProcessBabelTowerDataset:
    def __init__(self, dataset):
        """
        Initialize the class with the dataset.
        """
        self.dataset = dataset
        self.processed_dataset = dataset
        # Define the prompt template directly as data within the class.
        self.template = (
            "Title: Convert C/C++ Code to CUDA\n"
            "Description: I need assistance in converting the provided C/C++ code into CUDA for GPU parallelism.\n"
            "Steps:\n"
            "1. Convert the C/C++ code to CUDA, optimizing for performance and correctness.\n"
            "2. Generate sample inputs and expected outputs for testing.\n"
            "3. Provide a Makefile with build and run instructions (include any necessary dependencies).\n"
            "4. Structure the output in two sections: a 'code' section containing the CUDA code and commands, and an 'analysis' section discussing the changes.\n"
            "Input code to be converted:\n"
            "<code>\n"
        )


    def process(self, task = None) -> Dataset:
        """
        Process the dataset to extract relevant fields and format them for training.
        """
        if task == "train":
            self.processed_dataset = self.process_for_training()
        elif task == "inference":
            self.processed_dataset = self.process_for_inference()
        else:
            raise ValueError(f"Invalid task specified: {task}. Valid options are: 'train', 'inference'.")
        
        return self.processed_dataset

    def process_for_training(self):
        """
        Process the dataset for training by extracting the relevant fields and formatting them.
        """
        pass
        # if self.dataset is None:
        #     raise ValueError("Dataset is not provided.")

        # # Extracting 'input' and 'output' fields from the dataset
        # processed_data = []
        # for entry in self.dataset:
        #     if 'input' in entry and 'output' in entry:
        #         processed_data.append({
        #             'input': entry['input'],
        #             'output': entry['output']
        #         })

        # self.processed_dataset = processed_data
        # return self.processed_dataset

    def format_input(dataset):
        pass

    # Template is only used for validation and testing
    def apply_template(self, task = None):
        """
        Apply a chat template to the dataset.
        """
        # Implement the logic to apply the chat template to the dataset
        def replace_code(example):
            final_query = self.template.replace("<code>", example.get("code", ""))
            return {"final_query": final_query}
        
        new_dataset = self.dataset.map(replace_code)
        return new_dataset
    
    def process_for_inference(self):
        """
        Process the dataset for inference by applying the template.
        This method prepares the dataset for inference.
        """
        if self.dataset is None:
            raise ValueError("Dataset is not provided.")

        # Apply the template to format the input for inference
        processed_dataset = self.apply_template()
        
        return processed_dataset
    
class ProcessBabelTowerTestValDataset():
    """
    Class to process the BabelTower dataset specifically for testing and validation.
    """
    def __init__(self, dataset):
        self.dataset = dataset
                # Define the prompt template directly as data within the class.
        self.template = (
            "Title: Convert C/C++ Code to CUDA\n"
            "Description: I need assistance in converting the provided C/C++ code into CUDA for GPU parallelism.\n"
            "Steps:\n"
            "1. Convert the C/C++ code to CUDA, optimizing for performance and correctness.\n"
            "2. Generate sample inputs and expected outputs for testing.\n"
            "3. Provide a Makefile with build and run instructions (include any necessary dependencies).\n"
            "4. Structure the output in two sections within the tags: For translated code <code> </code> section containing the CUDA code and commands, and for ananlysis <analysis> </analysis> section discussing the changes.\n"
            "Input code to be converted:\n"
            "<code>\n"
        )
    

    def process_dataset_for_eval(self, ds) -> Dataset:
        """
        Process the dataset for validation.
        """
        if ds is None:
            raise ValueError("Dataset is not provided.")
        # Apply the template to format the input for inference
        def replace_code(example):
            final_query = self.template.replace("<code>", example.get("cpp", ""))
            return {"final_query": final_query}
        
        new_ds = ds.map(replace_code)

        return new_ds

    
def process_dataset_hf_format(
        cpp_path_test=None, 
        cuda_path_test=None,
        cpp_path_val=None,
        cuda_path_val=None, 
        save_dir=None):
    if cpp_path_test is None or cuda_path_test is None or cpp_path_val is None or cuda_path_val is None:
        raise ValueError("Path to the dataset file is not provided.")
    

    def read_and_parse_file(file_path):
        results = []
        cpp = []
        cuda = []
        try:
            with open(file_path[0], 'r') as f:
                # Read the first file to initialize the parsing
                for line in f:
                    if not line.strip():
                        continue
                    cpp_code = line.strip()
                    cpp.append(cpp_code)
            with open(file_path[1], 'r') as f:
                # Read the second file to initialize the parsing
                for line in f:
                    if not line.strip():
                        continue
                    cuda_code = line.strip()
                    cuda.append(cuda_code)
            # Ensure both lists are of the same length
            if len(cpp) != len(cuda):
                raise ValueError(
                    f"Mismatch in number of lines between {file_path[0]} and {file_path[1]}: "
                    f"{len(cpp)} (cpp) vs {len(cuda)} (cuda)."
                )
            # Combine the two lists into a list of tuples
            results = []
            for i in range(len(cpp)):
                # Append the tuple of (cpp_code, cuda_code)
                cpp_code = cpp[i]
                cuda_code = cuda[i]
                # Append the tuple (cpp_code, cuda_code) to results
                results.append((cpp_code, cuda_code))
            return results
        except Exception as e:
            raise RuntimeError(f"Error reading file {file_path}: {e}")
    
    file_path_test = [cpp_path_test, cuda_path_test]
    parsed_lines_test = read_and_parse_file(file_path_test)
    # Build lists of identifiers and codes
    cpp_test = [cpp_code for cpp_code, cuda_code in parsed_lines_test]
    cuda_test = [cuda_code for cpp_code, cuda_code in parsed_lines_test]


    file_path_val = [cpp_path_val, cuda_path_val]
    parsed_lines_val = read_and_parse_file(file_path_val)
    # Build lists of identifiers and codes
    cpp_val = [cpp_code for cpp_code, cuda_code in parsed_lines_val]
    cuda_val = [cuda_code for cpp_code, cuda_code in parsed_lines_val]
    # Create a Hugging Face Dataset from list of values
    #{"identifier": identifiers, "code": codes})
    ds_test = Dataset.from_dict(
        {
                "cpp": cpp_test,
                "cuda": cuda_test
        })
    ds_val = Dataset.from_dict(
        {
                "cpp": cpp_val,
                "cuda": cuda_val
        })
    ds = Dataset.from_dict(
        {
            "cpp": cpp_test + cpp_val,
            "cuda": cuda_test + cuda_val
        }
    )
    # Save the dataset to a JSON file
    if save_dir is not None:
        #ds_test.save_to_disk(save_dir)
        #ds_val.save_to_disk(save_dir)
        ds.save_to_disk(save_dir)

    return ds, ds_test, ds_val