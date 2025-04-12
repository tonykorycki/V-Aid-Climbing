import requests
from config import HF_API_TOKEN, LLM_API_URL

def test_api_connection():
    headers = {
        "Authorization": f"Bearer {HF_API_TOKEN}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "inputs": "Hello, how are you?",
        "parameters": {
            "max_new_tokens": 50,
            "return_full_text": False
        }
    }
    
    print(f"Testing connection to {LLM_API_URL}")
    print(f"Using token: {HF_API_TOKEN[:4]}...{HF_API_TOKEN[-4:]}")
    
    try:
        response = requests.post(LLM_API_URL, headers=headers, json=payload)
        print(f"Status code: {response.status_code}")
        print(f"Response: {response.text}")
        return response.status_code == 200
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    test_api_connection()

