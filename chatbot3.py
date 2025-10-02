from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List, Dict, Optional

class Chatbot:
    def __init__(self, context, model_name: str = "microsoft/DialoGPT-large", device: str = None):
        """
        Initialize the chatbot with a specified model.
        
        Args:
            model_name: Name of the pretrained model to use
            device: Hardware device to use ('cuda' or 'cpu')
        """
        self.model_name = model_name
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.conversation_history = []
        self.max_history = 5  # Keep last 5 exchanges (this isnt necessary i only added it for show but in reality if the front is good i wouldve removed it )
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name).to(self.device)
            
            print(f"Chatbot initialized with {self.model_name} on {self.device}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize chatbot: {str(e)}")

    def generate_response(self, user_input: str) -> str:
        """
        Generate a response to user input while maintaining conversation context.
        
        Args:
            user_input: The user's message
            
        Returns:
            The chatbot's response
        """
        try:
            #update conversation history
            self._update_history("User", user_input)
            
            prompt = self._build_prompt()
            
            #generate response (try changing the params )
            input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            output = self.model.generate(
                input_ids,
                max_new_tokens=150,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            #decoding and cleaning the response
            full_response = self.tokenizer.decode(output[0], skip_special_tokens=True)
            bot_response = self._extract_response(full_response, prompt)
            
            self._update_history("AI", bot_response)
            
            return bot_response
            
        except Exception as e:
            print(f"Error in generation: {str(e)}")
            return self._fallback_response(user_input)

    def _build_prompt(self) -> str:
        """Constructs a prompt from conversation history"""
        if not self.conversation_history:
            return ""
            
        prompt_lines = []
        for i, exchange in enumerate(self.conversation_history[-self.max_history:]):
            role, content = exchange
            prompt_lines.append(f"{role}: {content}")
            
        prompt_lines.append("AI:") 
        return "\n".join(prompt_lines)

    def _extract_response(self, full_response: str, prompt: str) -> str:
        """Extracts just the new response from the generated text"""
        response = full_response[len(prompt):].strip()
        
        response = response.split("User:")[0].split("AI:")[0].strip()
        
        #removing incomplete sentences
        if response and response[-1] not in {'.', '!', '?'}:
            last_punct = max(response.rfind('.'), response.rfind('!'), response.rfind('?'))
            if last_punct >= 0:
                response = response[:last_punct+1]
        
        return response

    def _update_history(self, role: str, content: str):
        """
        Update conversation history while maintaining max length.
        
        Args:
            role: Either 'User' or 'AI'
            content: The message content
        """
        self.conversation_history.append((role, content))
        
        #trim history if too long
        if len(self.conversation_history) > self.max_history * 2:
            self.conversation_history = self.conversation_history[-self.max_history * 2:]

    def _fallback_response(self, user_input: str) -> str:
        """Generate a fallback response when the model fails"""
        question_words = {'what', 'why', 'how', 'when', 'where', 'who', 'which', 'can', 'could'}
        first_word = user_input.strip().lower().split()[0] if user_input else ''
        
        if first_word in question_words:
            return "That's an interesting question. I need to think more about that."
        elif '?' in user_input:
            return "I'm not entirely sure about that. Could you rephrase or ask something else?"
        elif any(word in user_input.lower() for word in ['hi', 'hello', 'hey']):
            return "Hello! How can I help you today?"
        else:
            return "I see. Could you tell me more about that?"

    def reset_conversation(self):
        """Reset the conversation history"""
        self.conversation_history = []

