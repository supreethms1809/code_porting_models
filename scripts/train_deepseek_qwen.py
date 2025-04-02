import os
import sys
import yaml
from datasets import load_dataset

# Add the root directory of the project to the PYTHONPATH
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.logging import setup_log
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from datasets import load_from_disk, IterableDataset
from trl import SFTTrainer, SFTConfig
from transformers import DataCollatorForLanguageModeling, Trainer
logger = setup_log()
import torch
from peft import get_peft_model, LoraConfig, TaskType

device = "cuda" if torch.cuda.is_available() else "cpu"

save_directory = f"/home/sureshm/models/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

# Load the model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
#tokenizer.save_pretrained(save_directory)
#model.save_pretrained(save_directory)
config = model.config
# print(dir(model))
logger.info("Model and tokenizer loaded successfully.")

# Load dataset
# Define the dataset directory
dataset_dir = os.path.join(os.environ.get('BASE_DIR', '/home/sureshm/code_porting_models'), 'dataset')
dataset_path = os.path.join(dataset_dir, "code_dataset/V2combined_dataset")
ds_local = load_from_disk(dataset_path)

ds_local = ds_local.rename_column("content", "text")
subset_size = min(100, len(ds_local))  # Adjust this number based on your needs
train_dataset = ds_local.select(range(subset_size))
print(f"train dataset column names: {train_dataset.column_names}")

train_dataset = train_dataset.remove_columns([col for col in train_dataset.column_names if col != 'text'])
print(f"train dataset column names after removing: {train_dataset.column_names}")

#iterable_train_dataset = train_dataset.iter(batch_size=32)

datacollator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)
# train_dataset = IterableDataset.from_generator(lambda: iter(train_dataset),
#                                                features=train_dataset.features)

sft = False
ft = "sft" # lora
model_dir_name = ft + "_trained_model"
model_dir = os.path.join(os.environ.get('BASE_DIR', '/home/sureshm/code_porting_models'), model_dir_name)

if ft == "sft" and sft:
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
        dataset_num_proc=72
    )

    # Load TRL SFT Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=None,
        tokenizer=tokenizer,
        args=training_args,
        data_collator=datacollator
    )

    trainer.train()

lora = True
# LORA
ft = "lora"
if ft == "lora" and lora:
    tokenizer1 = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    model1 = AutoModelForCausalLM.from_pretrained("deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model1 = get_peft_model(model1, lora_config)
    model1.print_trainable_parameters()
    trainer = Trainer(
        model=model1,
        train_dataset=train_dataset,
        eval_dataset=None,
        tokenizer=tokenizer1,
        data_collator=datacollator
    )
    trainer.train()
