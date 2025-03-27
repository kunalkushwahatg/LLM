import torch
import torch.nn as nn
from transformers import PreTrainedModel
from custom_config import MyModelConfig

class MyModel(PreTrainedModel):
    config_class = MyModelConfig
    
    def __init__(self, config):
        super().__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([nn.Linear(config.hidden_size, config.hidden_size) for _ in range(config.num_layers)])
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        
    def forward(self, input_ids, labels=None):
        x = self.embedding(input_ids)
        for layer in self.layers:
            x = torch.relu(layer(x))
        logits = self.classifier(x[:, 0, :])  # Use [CLS] token for classification
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
        
        return {"loss": loss, "logits": logits} if loss is not None else {"logits": logits}

if __name__ == "__main__":
    config = MyModelConfig(hidden_size=128, num_layers=2, num_heads=2, num_labels=2)
    model = MyModel(config)
    input_ids = torch.randint(0, config.vocab_size, (1, 10))
    output = model(input_ids)
    print(output)
    # Output: {'loss': tensor(0.6931, grad_fn=<NllLossBackward>), 'logits': tensor([[-0.0197, -0.0147]], grad_fn=<AddmmBackward>)}