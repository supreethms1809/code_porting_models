from transformers import AutoTokenizer, AutoModelForCausalLM

# Specify the model name from the Hugging Face Hub
model_name = ["deepseek-ai/DeepSeek-R1-Distill-Llama-8B", 
              "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
              "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",]

for name in model_name:
    print(f"Downloading model {name} to /workspace/models/{name}")
    
    # Define the local directory to save the model and tokenizer
    save_directory = f"/workspace/models/{name}"
    
    # Download and save the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(name)
    tokenizer.save_pretrained(save_directory)

    # Download and save the model
    model = AutoModelForCausalLM.from_pretrained(name)
    model.save_pretrained(save_directory)
    print(f"Model {name} downloaded and saved to {save_directory}")