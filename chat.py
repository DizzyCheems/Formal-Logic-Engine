import ollama
import asyncio
import json
import os
from datetime import datetime
from sentence_transformers import SentenceTransformer, util
import numpy as np
import torch

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialize embedding model
try:
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Error loading embedding model: {e}")
    embedder = None

# File paths for memory storage
LONG_TERM_MEMORY_FILE = 'gemma3_long_term_memory.json'
SHORT_TERM_MEMORY_FILE = 'gemma3_short_term_memory.json'

def initialize_long_term_memory():
    if os.path.exists(LONG_TERM_MEMORY_FILE):
        try:
            with open(LONG_TERM_MEMORY_FILE, 'r', encoding='utf-8') as file:
                content = file.read().strip()
                if content:
                    memories = json.loads(content)
                    # Generate embeddings if missing
                    if embedder:
                        for memory in memories:
                            if not memory.get('embedding'):
                                memory['embedding'] = embedder.encode(
                                    memory['prompt'], convert_to_tensor=False
                                ).astype(np.float32).tolist()
                        # Save updated memories with embeddings
                        with open(LONG_TERM_MEMORY_FILE, 'w', encoding='utf-8') as file:
                            json.dump(memories, file, indent=2)
        except json.JSONDecodeError as e:
            print(f"Error reading {LONG_TERM_MEMORY_FILE}: {e}. Initializing empty.")
            memories = []
    else:
        # Initialize empty long-term memory file
        memories = []
        try:
            with open(LONG_TERM_MEMORY_FILE, 'w', encoding='utf-8') as file:
                json.dump(memories, file, indent=2)
        except Exception as e:
            print(f"Error initializing {LONG_TERM_MEMORY_FILE}: {e}")

def save_memory(prompt, response, memory_file, is_short_term=False):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    # Generate embeddings if embedder is available
    embedding = embedder.encode(prompt, convert_to_tensor=False).astype(np.float32).tolist() if embedder else []
    
    memory_entry = {
        'timestamp': timestamp,
        'prompt': prompt,
        'response': response,
        'embedding': embedding
    }
    
    # Load existing memories
    memories = []
    if os.path.exists(memory_file):
        try:
            with open(memory_file, 'r', encoding='utf-8') as file:
                content = file.read().strip()
                if content:
                    memories = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"Error reading {memory_file}: {e}. Starting fresh.")
    
    # For short-term memory, keep only the last 3 entries
    if is_short_term:
        memories.append(memory_entry)
        memories = memories[-3:]  # Keep only the last 3
    
    # Save to file
    try:
        with open(memory_file, 'w', encoding='utf-8') as file:
            json.dump(memories, file, indent=2)
    except Exception as e:
        print(f"Error writing to {memory_file}: {e}")

def load_relevant_memories(prompt, memory_file, top_k=3):
    if not os.path.exists(memory_file) or not embedder:
        return []
    
    try:
        with open(memory_file, 'r', encoding='utf-8') as file:
            content = file.read().strip()
            if not content:
                return []
            memories = json.loads(content)
    except json.JSONDecodeError as e:
        print(f"Error reading {memory_file}: {e}")
        return []
    
    if not memories:
        return []
    
    # Compute embedding for the input prompt
    prompt_embedding = embedder.encode(prompt, convert_to_tensor=False).astype(np.float32)
    
    # Compute similarity scores
    memory_embeddings = [np.array(memory['embedding'], dtype=np.float32) for memory in memories if memory.get('embedding')]
    if not memory_embeddings:
        return []
    
    # Convert to tensor for cosine similarity
    prompt_embedding_tensor = torch.tensor(prompt_embedding, dtype=torch.float32)
    memory_embeddings_tensor = torch.tensor(np.array(memory_embeddings), dtype=torch.float32)
    
    similarities = util.cos_sim(prompt_embedding_tensor, memory_embeddings_tensor)[0]
    top_indices = np.argsort(similarities.numpy())[-top_k:][::-1]
    
    # Return top_k relevant memories
    return [
        f"Previous prompt: {memories[i]['prompt']}\nPrevious response: {memories[i]['response']}"
        for i in top_indices
    ]

async def generate_response(input_seq, max_words=140):
    # Initialize long-term memory if it doesn't exist
    initialize_long_term_memory()
    
    # Load relevant memories
    long_term_memories = load_relevant_memories(input_seq, LONG_TERM_MEMORY_FILE, top_k=2)
    short_term_memories = load_relevant_memories(input_seq, SHORT_TERM_MEMORY_FILE, top_k=3)
    
    # Construct prompt with clear memory context
    memory_context = ""
    if long_term_memories or short_term_memories:
        memory_context = "Relevant past interactions:\n" + "\n\n".join(long_term_memories + short_term_memories) + "\n\n"
    
    # Emphasize current prompt and avoid repetitive questions
    prompt = f"{memory_context}Current user input: {input_seq}\nRespond directly to the current input, using the context if relevant. Do not repeat questions about preferences if they are already specified in the input or recent interactions."
    
    try:
        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: ollama.generate(model='gemma3', prompt=prompt)['response']
        )
        # Truncate to max_words
        words = response.split()
        if len(words) > max_words:
            response = ' '.join(words[:max_words]) + '...'
        
        # Save only to short-term memory
        save_memory(input_seq, response, SHORT_TERM_MEMORY_FILE, is_short_term=True)
        
        return response
    except Exception as e:
        print(f"Error in generate_response: {e}")
        return "An error occurred while processing your request. Please try again."

async def main():
    print("Gemma3 Interaction Interface with Memory")
    print("Type your prompt and press Enter. Type 'exit' to quit.")
    
    while True:
        user_input = input("\nYour prompt: ")
        if user_input.lower() == 'exit':
            print("Goodbye!")
            break
        
        response = await generate_response(user_input)
        print(f"\nGemma3: {response}")

if __name__ == "__main__":
    asyncio.run(main())