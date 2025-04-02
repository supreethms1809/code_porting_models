import os
import sys
import yaml
# Add the root directory of the project to the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.logging import setup_log

from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from datasets import Dataset
from datasets import load_from_disk
from transformers import DataCollatorForLanguageModeling, Trainer
from trl import SFTTrainer, SFTConfig

save_directory = f"/Users/ssuresh/aiml/models/deepseek-ai/DeepSeek-R1-Distill-Llama-8B"

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Llama-8B")

print("Model and tokenizer loaded successfully.")
# Load dataset
dataset_dir = os.path.join(os.environ.get('BASE_DIR', '/Users/ssuresh/aiml/code_porting_models/'), 'dataset')
ds = Dataset.load_from_disk(os.path.join(dataset_dir, 'babeltower'))
print(f"Dataset loaded successfully. {ds}")

ds = ds.rename_column("code", "text")
subset_size = min(100, len(ds))  # Adjust this number based on your needs
train_dataset = ds.select(range(subset_size))
print(f"train dataset column names: {train_dataset.column_names}")

train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if col != 'text'])
print(f"train dataset column names after removing: {train_dataset.column_names}")

datacollator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)
model_dir_name ="trained_model"
model_dir = os.path.join(os.environ.get('BASE_DIR', '/Users/ssuresh/aiml/code_porting_models/'), model_dir_name)

# Training arguments
training_args = SFTConfig(
    dataset_text_field="text",
    output_dir=model_dir,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=2e-5,
    logging_steps=100,
    save_steps=5000,
    max_seq_length=1024,
    overwrite_output_dir=True,
    dataset_batch_size=32,
    dataset_num_proc=16
)

# Load TRL SFT Trainer
trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=None,
    args=training_args,
    data_collator=datacollator
)

trainer.train()