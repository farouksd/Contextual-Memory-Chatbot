from transformers import pipeline, AutoTokenizer
import torch
import re

class Chatbot:
    def __init__(self):
        # Model selection - using a dialogue-optimized model
        #self.model_name = "microsoft/DialoGPT-medium"  
        self.model_name = "gpt2-xl" 
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            # Initialize tokenizer and model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.generator = pipeline(
                "text-generation",
                model=self.model_name,
                tokenizer=self.tokenizer,
                device=0 if self.device == "cuda" else -1
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize chatbot: {str(e)}")

    def generate_response(self, user_input, conversation_history=None):
        try:
            # Build the prompt with conversation context
            prompt = self._build_prompt(user_input, conversation_history)
            
            # Generate response with constrained parameters
            response = self.generator(
                prompt,
                max_new_tokens=100,
                num_return_sequences=1,
                temperature=0.8,  # Balanced between creative and focused
                top_p=0.9,
                repetition_penalty=1.2,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract and clean the response
            full_text = response[0]["generated_text"]
            bot_response = self._extract_response(full_text, prompt)
            
            return bot_response if bot_response else self._fallback_response(user_input)
            
        except Exception as e:
            print(f"Generation error: {str(e)}")
            return self._fallback_response(user_input)

    def _build_prompt(self, user_input, history=None):
        """Constructs a well-structured prompt for the model"""
        if history:
            return f"{history}\nUser: {user_input}\nAI:"
        return f"User: {user_input}\nAI:"

    def _extract_response(self, full_text, prompt):
        """Extracts just the AI's response from the generated text"""
        # Remove the prompt to get just the new response
        response = full_text[len(prompt):].strip()
        
        # Clean up any extra generated dialogue turns
        response = re.split(r'\nUser:|\nAI:', response)[0]
        
        # Remove any incomplete sentences at the end
        if response and response[-1] not in {'.', '!', '?'}:
            last_sentence_end = max(
                response.rfind('.'),
                response.rfind('!'),
                response.rfind('?')
            )
            if last_sentence_end >= 0:
                response = response[:last_sentence_end+1]
        
        return response.strip()

    def _fallback_response(self, user_input):
        """Generates a simple but relevant fallback response"""
        question_words = {'what', 'why', 'how', 'when', 'where', 'who', 'which'}
        first_word = user_input.strip().lower().split()[0] if user_input else ''
        
        if first_word in question_words:
            return "That's an interesting question. Let me think about that."
        elif '?' in user_input:
            return "I'm not entirely sure about that. What do you think?"
        else:
            return "Tell me more about that."