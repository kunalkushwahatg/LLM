from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
class Inference:
    def __init__(self, model_name,tokenizer_name, device='cpu'):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return outputs

    def generate(self, prompt, max_length=50):
        inputs = self.tokenizer(prompt, return_tensors='pt').to(self.device)
        with torch.no_grad():
            outputs = self.model.generate(inputs['input_ids'], max_length=max_length)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    

if __name__ == "__main__":
    config = json.load(open("./config/pretrain_config.json", "r"))
    model_name = config["model_name"]
    tokenizer_name = config["tokenizer_name"]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    inference = Inference(model_name,tokenizer_name, device)
    prompt = "Hello, how are you?"
    print(inference.generate(prompt))