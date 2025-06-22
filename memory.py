from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
import os

class ConversationMemory:
    def __init__(self, storage_file="memory.json"):
        self.storage_file = storage_file
        self.memory = []
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        
        if not os.path.exists(storage_file):
            self._initialize_empty_file()
        else:
            self.load_memory()
    
    def _initialize_empty_file(self):
        """Create a properly formatted empty JSON file"""
        with open(self.storage_file, 'w') as f:
            json.dump([], f)
    
    def add_conversation(self, user_input, bot_response):
        """Add a new conversation with proper embedding handling"""
        try:
            embedding = self._get_embedding(user_input + " " + bot_response)
            entry = {
                "user": user_input,
                "bot": bot_response,
                "embedding": embedding.tolist()  # Convert numpy array to list
            }
            self.memory.append(entry)
            self.save_memory()
        except Exception as e:
            print(f"Error adding conversation: {e}")
    
    def get_relevant_memories(self, query, top_k=3):
        """Get relevant memories with proper error handling"""
        if not self.memory:
            return []
            
        try:
            query_embedding = self._get_embedding(query)
            embeddings = [np.array(entry["embedding"]) for entry in self.memory]
            similarities = cosine_similarity([query_embedding], embeddings)[0]
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            return [self.memory[i] for i in top_indices]
        except Exception as e:
            print(f"Error retrieving memories: {e}")
            return []
    
    def _get_embedding(self, text):
        """Safe embedding generation"""
        return self.embedder.encode(text)
    
    def save_memory(self):
        """Save memory with proper error handling"""
        try:
            memory_to_save = []
            for entry in self.memory:
                new_entry = entry.copy()
                if "embedding" in new_entry and isinstance(new_entry["embedding"], np.ndarray):
                    new_entry["embedding"] = new_entry["embedding"].tolist()
                memory_to_save.append(new_entry)
            
            temp_file = self.storage_file + ".tmp"
            with open(temp_file, 'w') as f:
                json.dump(memory_to_save, f, indent=2)
            
            if os.path.exists(self.storage_file):
                os.replace(temp_file, self.storage_file)
            else:
                os.rename(temp_file, self.storage_file)
                
        except Exception as e:
            print(f"Error saving memory: {e}")
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except:
                    pass
    
    def load_memory(self):
        """Load memory with robust error handling"""
        try:
            if os.path.exists(self.storage_file):
                with open(self.storage_file, 'r') as f:
                    content = f.read()
                    if not content.strip():  
                        self.memory = []
                    else:
                        self.memory = json.loads(content)
            else:
                self.memory = []
        except json.JSONDecodeError:
            print("Corrupted memory file detected - resetting memory")
            self.memory = []
            self._initialize_empty_file()
        except Exception as e:
            print(f"Error loading memory: {e}")
            self.memory = []
    
    def clear_memory(self):
        """Completely reset the memory"""
        self.memory = []
        self._initialize_empty_file()
