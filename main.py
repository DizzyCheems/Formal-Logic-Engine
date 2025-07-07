import ollama
import asyncio
import json
from concurrent.futures import ThreadPoolExecutor
import os
from datetime import datetime

# File to store the mindmap
MINDMAP_FILE = "mindmap.json"

def initialize_mindmap():
    """Initialize or load the mindmap JSON file."""
    if not os.path.exists(MINDMAP_FILE):
        with open(MINDMAP_FILE, 'w') as f:
            json.dump({"nodes": [], "edges": []}, f, indent=2)
    with open(MINDMAP_FILE, 'r') as f:
        return json.load(f)

def update_mindmap(prompt, response):
    """Update the mindmap JSON with a new prompt-response pair."""
    mindmap = initialize_mindmap()
    timestamp = datetime.now().isoformat()
    
    # Add prompt and response as nodes
    prompt_node = {"id": f"prompt_{len(mindmap['nodes'])}", "label": prompt, "type": "prompt", "timestamp": timestamp}
    response_node = {"id": f"response_{len(mindmap['nodes'])+1}", "label": response, "type": "response", "timestamp": timestamp}
    
    # Add nodes and edge to mindmap
    mindmap['nodes'].extend([prompt_node, response_node])
    mindmap['edges'].append({"from": prompt_node['id'], "to": response_node['id'], "timestamp": timestamp})
    
    # Save updated mindmap
    with open(MINDMAP_FILE, 'w') as f:
        json.dump(mindmap, f, indent=2)

async def get_response(user_prompt):
    """
    Sends a user prompt to Gemma 3 via ollama and returns a response limited to 140 words.
    """
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor() as pool:
        response = await loop.run_in_executor(pool, lambda: ollama.chat(model="gemma3", messages=[{"role": "user", "content": user_prompt}]))
        refined_text = (response['message']['content'].replace('\n', ' ').strip() 
                        if 'message' in response and 'content' in response['message'] 
                        else "Sorry, I couldn't generate a response.")
        words = refined_text.split()
        return ' '.join(words[:140])

async def main():
    """
    Allows multiple prompts to Gemma 3 until the user types 'quit', updating the mindmap.
    """
    print("Enter your prompt for Gemma 3 (type 'quit' to exit):")
    while True:
        user_prompt = input("Prompt: ")
        if user_prompt.lower() == "quit":
            print("Goodbye!")
            break
        response = await get_response(user_prompt)
        print(f"Gemma 3: {response}\n")
        update_mindmap(user_prompt, response)

if __name__ == "__main__":
    asyncio.run(main())