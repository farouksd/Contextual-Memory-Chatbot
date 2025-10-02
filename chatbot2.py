from transformers import pipeline, AutoTokenizer

class Chatbot:
    def __init__(self):
        #"gpt2-xl" 
        self.model_name = "NousResearch/Hermes-2-Pro-Mistral-7B"  #consider using a larger model if needed
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.generator = pipeline(
            "text-generation",
            model=self.model_name,
            tokenizer=self.tokenizer,
            device="cpu"  # Change to "cuda" if using GPU
        )
    def generate_response(self, prompt, context=None):
        if context:
            full_prompt = f"Context: {context}\nUser: {prompt}\nBot:"
        else:
            full_prompt = f"User: {prompt}\nBot:"
        
        inputs = self.tokenizer(full_prompt, return_tensors="pt")
        input_length = inputs.input_ids.shape[1]
        
        #dynamic length adjustment
        max_new_tokens = min(100, 1024 - input_length)  
        try:
            response = self.generator(
                full_prompt,
                max_new_tokens=max_new_tokens,  
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                truncation=True
            ) 
            generated_text = response[0]["generated_text"]
            bot_response = generated_text.split("Bot:")[-1].strip()
            return bot_response
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I encountered an error while generating a response. Please try again."

