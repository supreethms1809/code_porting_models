# class to process the BabelTower dataset

class ProcessBigStackTDataset:
    def __init__(self, dataset):
        """
        Initialize the class with the dataset.
        """
        self.dataset = dataset
        self.processed_dataset = None
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


    def process(self, task = None):
        """
        Process the dataset to extract relevant fields and format them for training.
        """
        if task == "train":
            self.processed_dataset = self.process_for_training()
        elif task == "eval":
            self.processed_dataset = self.process_dataset_for_evaluation()
        elif task == "test":
            self.processed_dataset = self.process_dataset_for_testing()
        
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
    
    def process_dataset_for_evaluation(self):
        """
        Process the dataset specifically for evaluation purposes.
        This method can be customized based on evaluation needs.
        """
        pass
        # if self.dataset is None:
        #     raise ValueError("Dataset is not provided.")

        # # Extracting 'text' field from the dataset for evaluation
        # processed_data = []
        # for entry in self.dataset:
        #     if 'text' in entry:
        #         processed_data.append(entry['text'])

        # self.processed_dataset = processed_data
        # return self.processed_dataset

    def process_dataset_for_testing(self):
        """
        Process the dataset specifically for testing purposes.
        This method can be customized based on testing needs.
        """
        pass
        # if self.dataset is None:
        #     raise ValueError("Dataset is not provided.")

        # # Extracting 'text' field from the dataset for testing
        # processed_data = []
        # for entry in self.dataset:
        #     if 'text' in entry:
        #         processed_data.append(entry['text'])

        # self.processed_dataset = processed_data
        # return self.processed_dataset

    def format_input(self, dataset):
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