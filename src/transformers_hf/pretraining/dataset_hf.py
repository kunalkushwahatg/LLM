import json
from datasets import load_dataset, Dataset, DatasetDict
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from itertools import chain
class LLMPretrainDataset:
    def __init__(self, config_path):
        """Initializes the dataset for LLM pretraining."""
        
        with open(config_path, "r") as f:
            self.config = json.load(f)

        self.tokenizer = AutoTokenizer.from_pretrained(self.config["tokenizer_name"])
        
        self.tokenizer.pad_token = self.tokenizer.eos_token

        self.max_length = self.config["max_length"]
        self.debug_mode = self.config.get("debug", False)
        self.debug_ratio = self.config.get("debug_ratio", 0.1)
        self.chunk_size = self.config.get("chunk_size", 1024)

        # Load dataset
        self.dataset = self.load_dataset()

        if self.debug_mode:
            self.dataset = self.dataset.shuffle().select(range(int(len(self.dataset) * self.debug_ratio)))

        # Split dataset
        self.dataset = self.dataset.train_test_split(test_size=self.config["validation_ratio"])

        #tokenize the dataset
        self.dataset = self.dataset.map(self.tokenize_function,batched=True,remove_columns='text')
        #concatenate the tokens
        self.dataset = self.dataset.map(self.concat,batched=True,batch_size=1000000)

        #chunk the tokens
        self.dataset = self.dataset.map(self.chunk,batched=True,batch_size=2)


    def concat(self,examples):    
        examples["input_ids"]=[list(chain.from_iterable(examples['input_ids']))] # convert chain to list of tokens
        examples["attention_mask"]=[list(chain.from_iterable(examples['attention_mask']))] # convert chain to list of tokens
        if "token_type_ids" in examples:
            examples["token_type_ids"] = [list(chain.from_iterable(examples['token_type_ids']))]
    
        return examples

    def chunk(self,examples):    
        input_ids = examples["input_ids"][0] # List[List], pass the inner list      
        attention_mask = examples["attention_mask"][0] # List[List]
        input_ids_truncated = []
        attention_mask_truncated = []
        
        #slice with step_size=chunk_size
        for i in range(0,len(input_ids),self.chunk_size):
            chunk = input_ids[i:i+self.chunk_size]
            if len(chunk)==self.chunk_size: # drop the last chunk if not equal
                input_ids_truncated.append(chunk)
                attention_mask_truncated.append(attention_mask[i:i+self.chunk_size])     
        examples['input_ids']=input_ids_truncated
        examples["attention_mask"]=attention_mask_truncated
            
        return examples  

    def tokenize_function(self,example):
        return self.tokenizer(text=example["text"])

    def load_dataset(self):
        """Loads dataset from Hugging Face or local files."""
        dataset_source = self.config["preprocessed_dataset_path"]

        if dataset_source.startswith("hf://"):
            dataset_name = dataset_source.replace("hf://", "")
            dataset = load_dataset(dataset_name)["train"]
        elif dataset_source.endswith(".txt"):
            dataset = load_dataset("text", data_files=dataset_source)["train"]
        elif dataset_source.endswith(".csv"):
            dataset = load_dataset("csv", data_files=dataset_source)["train"]
        elif dataset_source.endswith(".json"):
            dataset = load_dataset("json", data_files=dataset_source)["train"]
        elif dataset_source.endswith(".parquet"):
            dataset = load_dataset("parquet", data_files=dataset_source)["train"]
        else:
            raise ValueError("Unsupported dataset format!")
        print(dataset.column_names)
        print(len(dataset))
        return dataset

    def save_dataset(self, save_dir):
        """Saves the dataset to the specified directory."""
        self.dataset.save_to_disk(save_dir)
        print(f"Dataset saved successfully to {save_dir}")

    def save_tokenizer(self, save_dir):
        """Saves the tokenizer to the specified directory."""
        self.tokenizer.save_pretrained(save_dir)
        print(f"Tokenizer saved successfully to {save_dir}")

if __name__ == "__main__":
    dataset = LLMPretrainDataset("./config/pretrain_config.json")
    dataset.save_tokenizer("./models/"+dataset.config["tokenizer_name"])
    dataset.save_dataset("./data/"+dataset.config["dataset_name"])
    

    #decoded tokens
    for i in range(1):
        print(dataset.tokenizer.decode(dataset.dataset['train']['input_ids'][i]))
        


