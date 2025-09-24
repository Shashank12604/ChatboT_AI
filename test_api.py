import requests
import json

# Test the API
try:
    response = requests.post(
        "http://127.0.0.1:8000/chat",
        json={"messages": [{"role": "user", "content": "Hello"}]},
        timeout=10
    )
    print(f"Status: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
