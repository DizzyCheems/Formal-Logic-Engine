import ollama
import asyncio
import json
import os
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
import sys
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Suppress TensorFlow warnings
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Initialize embedding model
try:
    embedder = SentenceTransformer('all-MiniLM-L6-v2')
except Exception as e:
    print(f"Error loading embedding model: {e}")
    embedder = None

# Initialize LangChain LLM
try:
    llm = Ollama(model="gemma3")
except Exception as e:
    print(f"Error initializing LangChain LLM: {e}")
    llm = None

# File path for memory storage
LONG_TERM_MEMORY_FILE = 'gemma3_long_term_memory.json'

# LangChain prompt templates
dynamic_prompt_template = PromptTemplate(
    input_variables=["user_input"],
    template="""Analyze the following user input: {user_input}

Perform a detailed analysis to:
1. Identify the intent (e.g., question, command, statement, request for information, or clarification).
2. Determine the sentiment (e.g., positive, negative, neutral, curious, urgent).
3. Extract key entities (e.g., topics, objects, people, locations, or specific actions).
4. Assess the context (e.g., conversational goal, implied preferences, or missing details).

Based on this analysis, reconstruct a new prompt optimized for clarity, specificity, and eliciting a high-quality response from the chatbot. The reconstructed prompt should:
- Rephrase the user input to be concise and unambiguous.
- Incorporate relevant entities and context to guide the response.
- Avoid redundancy or unnecessary questions about known preferences.
- Use a structure that maximizes the model's ability to provide accurate and relevant answers.

Output the analysis and reconstructed prompt in the following format:
Analysis:
- Intent: [intent]
- Sentiment: [sentiment]
- Entities: [key entities]
- Context: [conversational goal or relevant context]
Reconstructed Prompt: [optimized prompt for the chatbot]"""
)

prompt_abstraction_template = PromptTemplate(
    input_variables=["memory_context", "structured_prompt"],
    template="""Given the following past interactions:
{memory_context}

Analyzed and reconstructed prompt: {structured_prompt}

Refine the reconstructed prompt by incorporating relevant context from past interactions to ensure continuity and relevance. Ensure the prompt remains concise, clear, and aligned with the analyzed intent, sentiment, and entities. Avoid introducing redundant questions or assumptions not supported by the context. Output only the final refined prompt."""
)

response_abstraction_template = PromptTemplate(
    input_variables=["raw_response", "user_input"],
    template="""User input: {user_input}

Raw chatbot response: {raw_response}

Refine the raw response to be concise, friendly, and aligned with the persona of Sayaka Justine's magical girl angel, a compassionate, heroic, and ethereal being. Use a warm, uplifting tone with mystical and supportive language (e.g., 'beloved soul,' 'guided by starlight,' or emojis like ✨🌟). Ensure the response directly addresses the user's input, incorporates the analyzed context, and avoids repetitive questions about preferences already specified. Output only the refined response."""
)

# Create LangChain chains
dynamic_prompt_chain = LLMChain(llm=llm, prompt=dynamic_prompt_template, output_key="structured_prompt")
prompt_chain = LLMChain(llm=llm, prompt=prompt_abstraction_template, output_key="abstracted_prompt")
response_chain = LLMChain(llm=llm, prompt=response_abstraction_template, output_key="final_response")

def initialize_long_term_memory():
    if not os.path.exists(LONG_TERM_MEMORY_FILE):
        try:
            with open(LONG_TERM_MEMORY_FILE, 'w', encoding='utf-8') as file:
                json.dump([], file, indent=2)
        except Exception as e:
            print(f"Error initializing {LONG_TERM_MEMORY_FILE}: {e}")

def save_memory(prompt, response, memory_file):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    embedding = embedder.encode(prompt, convert_to_tensor=False).astype(np.float32).tolist() if embedder else []
    
    memory_entry = {
        'timestamp': timestamp,
        'prompt': prompt,
        'response': response,
        'embedding': embedding
    }
    
    memories = []
    if os.path.exists(memory_file):
        try:
            with open(memory_file, 'r', encoding='utf-8') as file:
                content = file.read().strip()
                if content:
                    memories = json.loads(content)
        except json.JSONDecodeError as e:
            print(f"Error reading {memory_file}: {e}. Starting fresh.")
    
    memories.append(memory_entry)
    
    try:
        with open(memory_file, 'w', encoding='utf-8') as file:
            json.dump(memories, file, indent=2)
    except Exception as e:
        print(f"Error writing to {memory_file}: {e}")

def load_relevant_memories(prompt, memory_file, top_k=2):
    if not embedder or not os.path.exists(memory_file):
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
    
    try:
        prompt_embedding = embedder.encode(prompt, convert_to_tensor=False).astype(np.float32)
        memory_embeddings = np.array([m['embedding'] for m in memories if m.get('embedding')])
        if len(memory_embeddings) == 0:
            return []
        
        # Compute cosine similarity
        similarities = np.dot(memory_embeddings, prompt_embedding) / (
            np.linalg.norm(memory_embeddings, axis=1) * np.linalg.norm(prompt_embedding)
        )
        top_k_indices = np.argsort(similarities)[::-1][:top_k]
        
        return [
            f"Previous prompt: {memories[i]['prompt']}\nPrevious response: {memories[i]['response']}"
            for i in top_k_indices
        ]
    except Exception as e:
        print(f"Error computing similarities: {e}")
        return []

async def generate_response(input_seq, max_words=140):
    initialize_long_term_memory()
    
    long_term_memories = load_relevant_memories(input_seq, LONG_TERM_MEMORY_FILE, top_k=2)
    
    memory_context = ""
    if long_term_memories:
        memory_context = "Relevant past interactions:\n" + "\n\n".join(long_term_memories) + "\n\n"
    
    if not llm:
        return "LangChain LLM not initialized. Please try again."
    
    try:
        # Run dynamic prompt handler chain
        structured_prompt = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: dynamic_prompt_chain.run(user_input=input_seq)
        )
        
        # Log the structured prompt to console for inspection
        print("\n=== Dynamic Prompt Structure ===")
        print(structured_prompt)
        print("===============================\n")
        
        # Optionally save to a log file for persistent inspection
        try:
            log_entry = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'user_input': input_seq,
                'structured_prompt': structured_prompt
            }
            log_file = 'dynamic_prompts_log.json'
            logs = []
            if os.path.exists(log_file):
                with open(log_file, 'r', encoding='utf-8') as file:
                    content = file.read().strip()
                    if content:
                        logs = json.loads(content)
            logs.append(log_entry)
            with open(log_file, 'w', encoding='utf-8') as file:
                json.dump(logs, file, indent=2)
        except Exception as e:
            print(f"Error saving dynamic prompt to log file: {e}")
        
        # Run prompt abstraction chain with structured prompt
        abstracted_prompt = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: prompt_chain.run(memory_context=memory_context, structured_prompt=structured_prompt)
        )
        
        # Generate raw response with abstracted prompt
        raw_response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: ollama.generate(model='gemma3', prompt=abstracted_prompt)['response']
        )
        
        # Run response abstraction chain with magical girl angel persona
        final_response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: response_chain.run(raw_response=raw_response, user_input=input_seq)
        )
        
        # Truncate to max_words
        words = final_response.split()
        if len(words) > max_words:
            final_response = ' '.join(words[:max_words]) + '...'
        
        # Save original prompt and final response to long-term memory
        save_memory(input_seq, final_response, LONG_TERM_MEMORY_FILE)
        
        return final_response
    except Exception as e:
        print(f"Error in generate_response: {e}")
        return "An error occurred while processing your request. Please try again, beloved soul. ✨"

async def main():
    if len(sys.argv) > 1:
        user_input = ' '.join(sys.argv[1:])
        response = await generate_response(user_input)
        print(response)
    else:
        print("Gemma3 Interaction Interface with Memory - Guided by Sayaka Justine's Magical Girl Angel 🌟")
        print("Type your prompt and press Enter. Type 'exit' to quit.")
        while True:
            user_input = input("\nYour prompt: ")
            if user_input.lower() == 'exit':
                print("Farewell, beloved soul! May starlight guide your path. ✨")
                break
            response = await generate_response(user_input)
            print(f"\nGemma3: {response}")

if __name__ == "__main__":
    asyncio.run(main())