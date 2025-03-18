""" 
curl http://localhost:11434/api/generate -d '{ "model": "llama3.2", "prompt": "How are you today?"}'
curl http://localhost:11434/api/generate -d '{ "model": "llama3.2", "prompt": "How are you today?", "stream": false}'"
"""
url = "http://localhost:11434/api/generate"
data = {
    "model": "deepseek-r1:8b",
    "prompt": "WHow are you today?"
}
import requests
import json
#response = requests.post(url, json=data)
#print(response.json())

response = requests.post(url, json=data, stream=True)

for line in response.iter_lines():
    if line:
        decoded_line = line.decode('utf-8')
        print(json.loads(decoded_line)["response"], end="", flush=True)