from transformers import TrainerCallback

class TextGenerationCallback(TrainerCallback):
    def __init__(self, tokenizer, model, num_samples=3):
        self.tokenizer = tokenizer
        self.model = model
        self.num_samples = num_samples

    def on_evaluate(self, args, state, control, **kwargs):
        print("\n### Text Generation for Verification ###")
        prompts = ["Once upon a time,", "The future of AI", "In a small village,"]
        for i, prompt in enumerate(prompts[:self.num_samples]):
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(**inputs, max_length=50, num_return_sequences=1)
            print(f"Prompt: {prompt}")
            print(f"Generated: {self.tokenizer.decode(outputs[0], skip_special_tokens=True)}\n")
