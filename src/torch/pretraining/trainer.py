import training_config

class Trainer:
    def __init__(self, model, train_loader, val_loader, config, device):
        self.model = model.to(device)
        self.optimizer =  training_config.get_optimizer(self.model.parameters())
        self.scheduler = training_config.get_scheduler(self.optimizer)
        self.scaler = torch.cuda.amp.GradScaler()  # For mixed precision
        

    def train_step(self, batch):
        x, y = batch
        with torch.cuda.amp.autocast():
            logits = self.model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()
        return loss.item()

    def train(self):
        for epoch in range(config.num_epochs):
            for batch in train_loader:
                loss = self.train_step(batch)
                
