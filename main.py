from flask import Flask, request, jsonify, send_from_directory
import requests
import json
import os

app = Flask(__name__)

@app.route('/')
def serve_ui():
    """Serve the web UI."""
    static_folder = os.path.join(os.getcwd(), 'static')
    index_path = os.path.join(static_folder, 'index.html')
    print(f"Looking for index.html at: {index_path}")  # Debug log
    if not os.path.exists(index_path):
        print(f"Error: index.html not found at {index_path}")
        return jsonify({"error": "index.html not found in static folder"}), 404
    return app.send_static_file('index.html')

@app.route('/chat', methods=['POST'])
def proxy_chat():
    """Proxy requests to Ollama API and handle streaming response."""
    try:
        data = request.json
        response = requests.post('http://localhost:11434/api/chat', json=data, stream=True, timeout=30)
        if response.status_code != 200:
            return jsonify({"error": f"Ollama API error: {response.status_code} - {response.text}"}), response.status_code
        
        # Collect streaming response
        full_response = ""
        for line in response.iter_lines():
            if line:
                try:
                    json_data = json.loads(line.decode('utf-8'))
                    if json_data.get('message') and json_data['message'].get('content'):
                        full_response += json_data['message']['content']
                except json.JSONDecodeError:
                    continue  # Skip invalid JSON lines
        return jsonify({"message": {"content": full_response}})
    except requests.exceptions.RequestException as e:
        return jsonify({"error": f"Failed to connect to Ollama server: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)