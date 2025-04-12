import requests
import numpy as np
from typing import Optional
from config import HF_API_TOKEN

def generate_route_description(grid_map: np.ndarray, difficulty: Optional[str] = None, 
                              use_local_llm: bool = False, api_url: Optional[str] = None) -> str:
    """Generate a description of the climbing route using Mistral API"""
    # Convert grid to string
    grid_str = ""
    for row in grid_map:
        grid_str += "".join(map(str, row)) + "\n"
    
    # Create prompt for the LLM
    prompt = f"""
You are a professional climbing route setter. Analyze this 12x12 grid representing a climbing wall.
In this grid:
- 0 represents empty space (no holds)
- 1 represents a small hold
- 2 represents a large hold or volume

The bottom of the grid is the start, and the top is the end of the route.
Here is the grid map:

{grid_str}

{"The predicted difficulty is " + difficulty if difficulty else ""}

Provide a concise but informative description of this climbing route. Include:
1. The overall flow/pattern of the route
2. Any notable features (like long reaches, crux sections, rest positions)
3. The approximate difficulty based on hold density and positioning
4. Any recommendations for climbers attempting this route

Keep your response under 150 words.
"""

    if api_url:
        # Prepare the API request
        headers = {
            "Authorization": f"Bearer {HF_API_TOKEN}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 300,
                "temperature": 0.7,
                "return_full_text": False
            }
        }
        
        # Send request to the API
        try:
            print(f"Sending request to Mistral API...")
            response = requests.post(api_url, headers=headers, json=payload)
            
            if response.status_code == 200:
                result = response.json()
                if isinstance(result, list) and len(result) > 0:
                    return result[0].get("generated_text", "No text generated")
                return str(result)
            else:
                print(f"API error: {response.status_code} - {response.text}")
                return generate_generic_description(grid_map, difficulty)
                
        except Exception as e:
            print(f"Error using API LLM: {e}")
            return generate_generic_description(grid_map, difficulty)
    else:
        return generate_generic_description(grid_map, difficulty)

def generate_generic_description(grid_map: np.ndarray, difficulty: Optional[str] = None) -> str:
    """Generate a fallback description if the API fails"""
    # Simple analysis of the route
    num_holds = np.sum(grid_map > 0)
    return f"This climbing route contains {num_holds} holds or volumes. The route appears to be of {difficulty if difficulty else 'moderate'} difficulty."