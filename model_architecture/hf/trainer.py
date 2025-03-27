from transformers import Trainer, TrainingArguments
from model import MyModel
from custom_config import MyModelConfig
from datasets import load_dataset

train_dataset = load_dataset("imdb")["train"]
eval_dataset = load_dataset("imdb")["test"]

tokenizer = MyModelConfig.tokenizer.from_pretrained("bert-base-uncased")


config = MyModelConfig(vocab_size=30000, num_labels=2)
model = MyModel(config)

training_args = TrainingArguments(
    output_dir="./results",
    num_train_epochs=3,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=64,
    learning_rate=5e-5,
    weight_decay=0.01,
    logging_dir="./logs",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

trainer.train()
