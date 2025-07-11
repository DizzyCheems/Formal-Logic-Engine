import ollama
import asyncio
import json
import os
from datetime import datetime
import numpy as np
from sentence_transformers import SentenceTransformer
import chromadb
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

# Initialize Chroma client (persistent storage)
try:
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
except Exception as e:
    print(f"Error initializing Chroma: {e}")
    chroma_client = None

# Initialize LangChain LLM
try:
    llm = Ollama(model="gemma3")
except Exception as e:
    print(f"Error initializing LangChain LLM: {e}")
    llm = None

# File path for memory storage
LONG_TERM_MEMORY_FILE = 'gemma3_long_term_memory.json'

# Chroma collection name
LONG_TERM_COLLECTION = 'gemma3-long-term'

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

Refine the raw response to be concise, friendly, and aligned with a warm, affectionate tone (e.g., using terms like 'sweet knight' or emojis like 😊🍦). Ensure the response directly addresses the user's input, incorporates the analyzed context, and avoids repetitive questions about preferences already specified. Output only the refined response."""
)

# Create LangChain chains
dynamic_prompt_chain = LLMChain(llm=llm, prompt=dynamic_prompt_template, output_key="structured_prompt")
prompt_chain = LLMChain(llm=llm, prompt=prompt_abstraction_template, output_key="abstracted_prompt")
response_chain = LLMChain(llm=llm, prompt=response_abstraction_template, output_key="final_response")

def initialize_chroma_collections():
    if not chroma_client:
        print("Chroma client not initialized. Skipping collection creation.")
        return
    
    try:
        chroma_client.get_or_create_collection(name=LONG_TERM_COLLECTION)
    except Exception as e:
        print(f"Error initializing Chroma collection: {e}")

def initialize_long_term_memory():
    if not os.path.exists(LONG_TERM_MEMORY_FILE):
        try:
            with open(LONG_TERM_MEMORY_FILE, 'w', encoding='utf-8') as file:
                json.dump([], file, indent=2)
        except Exception as e:
            print(f"Error initializing {LONG_TERM_MEMORY_FILE}: {e}")
        return
    
    try:
        with open(LONG_TERM_MEMORY_FILE, 'r', encoding='utf-8') as file:
            content = file.read().strip()
            memories = json.loads(content) if content else []
    except json.JSONDecodeError as e:
        print(f"Error reading {LONG_TERM_MEMORY_FILE}: {e}. Initializing empty.")
        memories = []
    
    if not chroma_client or not memories or not embedder:
        return
    
    collection = chroma_client.get_collection(LONG_TERM_COLLECTION)
    vectors = []
    for i, memory in enumerate(memories):
        if not memory.get('embedding'):
            try:
                embedding = embedder.encode(memory['prompt'], convert_to_tensor=False).astype(np.float32).tolist()
                memory['embedding'] = embedding
                vectors.append({
                    'id': f'lt_{i}',
                    'embedding': embedding,
                    'metadata': {'prompt': memory['prompt'], 'response': memory['response'], 'timestamp': memory['timestamp']},
                    'document': memory['prompt']
                })
            except Exception as e:
                print(f"Error generating embedding for memory {i}: {e}")
                memory['embedding'] = []
    
    if vectors:
        try:
            collection.upsert(
                ids=[v['id'] for v in vectors],
                embeddings=[v['embedding'] for v in vectors],
                metadatas=[v['metadata'] for v in vectors],
                documents=[v['document'] for v in vectors]
            )
            with open(LONG_TERM_MEMORY_FILE, 'w', encoding='utf-8') as file:
                json.dump(memories, file, indent=2)
        except Exception as e:
            print(f"Error upserting to Chroma {LONG_TERM_COLLECTION}: {e}")

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
    
    if chroma_client and embedder and embedding:
        collection = chroma_client.get_collection(LONG_TERM_COLLECTION)
        try:
            collection.upsert(
                ids=[f'lt_{len(memories)-1}'],
                embeddings=[embedding],
                metadatas=[{'prompt': prompt, 'response': response, 'timestamp': timestamp}],
                documents=[prompt]
            )
        except Exception as e:
            print(f"Error upserting to {LONG_TERM_COLLECTION}: {e}")
    
    try:
        with open(memory_file, 'w', encoding='utf-8') as file:
            json.dump(memories, file, indent=2)
    except Exception as e:
        print(f"Error writing to {memory_file}: {e}")

def load_relevant_memories(prompt, collection_name, memory_file, top_k=2):
    if not chroma_client or not embedder or not os.path.exists(memory_file):
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
        collection = chroma_client.get_collection(collection_name)
        query_results = collection.query(query_embeddings=[prompt_embedding.tolist()], n_results=top_k, include=['metadatas', 'documents'])
        matches = query_results['metadatas'][0]
        return [
            f"Previous prompt: {match['prompt']}\nPrevious response: {match['response']}"
            for match in matches
        ]
    except Exception as e:
        print(f"Error querying Chroma {collection_name}: {e}")
        return []

async def generate_response(input_seq, max_words=140):
    initialize_chroma_collections()
    initialize_long_term_memory()
    
    long_term_memories = load_relevant_memories(input_seq, LONG_TERM_COLLECTION, LONG_TERM_MEMORY_FILE, top_k=2)
    
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
        
        # Run response abstraction chain
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
        return "An error occurred while processing your request. Please try again."

async def main():
    if len(sys.argv) > 1:
        user_input = ' '.join(sys.argv[1:])
        response = await generate_response(user_input)
        print(response)
    else:
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