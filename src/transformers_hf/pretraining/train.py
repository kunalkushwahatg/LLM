import torch
from transformers import Trainer, TrainingArguments
from model_hf import GPT2Model 
from datasets import load_from_disk, DatasetDict
from transformers import AutoTokenizer
import json
from transformers import DataCollatorForLanguageModeling
from training_callback import TextGenerationCallback

config_path = "./config/pretrain_config.json"
model_path = "./models/"
tokenizer_path = "./tokenizers/"



# Load pretraining configuration
with open(config_path, "r") as f:
    config = json.load(f)

#pretrainin type
pretraining_type = config["pretraining_type"]

# Load preprocessed dataset from disk
dataset_load = DatasetDict({
    "train": load_from_disk("./data/"+config['dataset_name']+"/train"),
    "test": load_from_disk("./data/"+config['dataset_name']+"/test")
})



#load the tokenizer
tokenizer = AutoTokenizer.from_pretrained(config["tokenizer_name"])
tokenizer.pad_token = tokenizer.eos_token

# DataCollator for pretraining
if pretraining_type == "mlm":
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=config["mlm_probability"]
    )
elif pretraining_type == "clm":
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False, mlm_probability=0.0
    )
else:
    raise ValueError("Pretraining type not supported. Use 'mlm' or 'clm'.")

# Load model
model = GPT2Model().load_model()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
print("running on : ",device)

#test model with a sample input
input_text = "Hello, how are you?"
input_ids = tokenizer(input_text, return_tensors="pt").input_ids
output = model.generate(input_ids.to(device))
print(tokenizer.decode(output[0], skip_special_tokens=True))

model.train()

# Define training arguments
training_args = TrainingArguments(
    output_dir='./src/pretraining/checkpoints/'+config['model_name'],  # Directory to save model checkpoints
    evaluation_strategy="steps",  # Evaluate model at regular intervals
    eval_steps=200,  # Perform evaluation every 500 steps
    num_train_epochs=1,  # Number of training epochs
    per_device_train_batch_size=8,  # Batch size per GPU for training
    per_device_eval_batch_size=8,  # Batch size per GPU for evaluation
    learning_rate=2.5e-4,  # Initial learning rate
    lr_scheduler_type='cosine',  # Use cosine learning rate scheduler
    warmup_ratio=0.05,  # Percentage of total steps for warmup
    adam_beta1=0.9,  # Adam optimizer beta1 value
    adam_beta2=0.999,  # Adam optimizer beta2 value
    weight_decay=0.01,  # Weight decay for regularization
    logging_strategy="steps",  # Log training progress at intervals
    logging_steps=500,  # Log every 500 steps
    save_steps=5000,  # Save model checkpoint every 5000 steps
    save_total_limit=10,  # Keep only the last 10 checkpoints
    #report_to='wandb',  # Report training logs to Weights & Biases
)

# Initialize Hugging Face Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    tokenizer=tokenizer,
    data_collator=data_collator,
    train_dataset=dataset_load["train"],
    eval_dataset=dataset_load["test"],
    callbacks=[TextGenerationCallback(tokenizer, model)],
)

# Start training
trainer.train()

# Save the trained model and tokenizer
model.save_pretrained("./models/"+config['model_name'])
