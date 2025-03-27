from transformers import AutoModelForCausalLM   , AutoTokenizer
import sys
import os 

sys.path.append(os.path.abspath("./"))

class GPT2Model:
    def __init__(self):
        self.model = None
    def load_model(self):
        self.model = AutoModelForCausalLM.from_pretrained("gpt2")
        return self.model

    
if __name__ == "__main__":
    gpt2 = GPT2Model()
    model = gpt2.load_model()

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    input_text = "The is an  in Agra, Uttar Pradesh,  fifth Mughal emperor, Shah Jahan (r. 1628–1658) to house the tomb of his beloved wife, Mumtaz Mahal; it also houses the tomb of Shah Jahan himself. The tomb is the centrepiece of a 17-hectare (42-acre) complex, which includes a mosque and a guest house, and is set in formal gardens bounded on three sides by a crenellated wall."
    input_ids = tokenizer(input_text, return_tensors="pt").input_ids

    output = model.generate(input_ids,max_length=200)  
    print(tokenizer.decode(output[0], skip_special_tokens=True))

