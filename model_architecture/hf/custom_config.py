from transformers import PretrainedConfig

class MyModelConfig(PretrainedConfig):
    model_type = "my_model"
    
    def __init__(self, hidden_size=768, num_layers=12, num_heads=12, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.vocab_size = 50257
        self.tokenizer = "bert-base-uncased"
        


