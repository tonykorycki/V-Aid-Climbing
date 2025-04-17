import requests
import json
import numpy as np
from typing import Dict, Optional
from paths import HF_API_TOKEN  # <-- Add this import

def generate_route_description(grid_map: np.ndarray, 
                               difficulty: Optional[str] = None, 
                               use_local_llm: bool = False, 
                               api_url: Optional[str] = None) -> str:
    """
    Generate a natural language description of the climbing route based on the grid map.
    
    Args:
        grid_map: 12x12 numpy array where 0=no hold, 1=small hold, 2=large hold/volume
        difficulty: Predicted difficulty level of the route
        use_local_llm: Whether to use a local LLM (not recommended on Pi)
        api_url: URL for API-based LLM
        
    Returns:
        str: Natural language description of the climbing route
    """
    # Convert grid to string representation for the prompt
    grid_str = ""
    for row in grid_map:
        grid_str += "".join(map(str, row)) + "\n"
    
    # Create a carefully designed prompt for the LLM
    prompt = f"""[INST]
You are a professional climbing route setter. Analyze this 12x12 grid representing a climbing wall.
In this grid:
- 0 represents empty space (no holds)
- 1 represents a small hold
- 2 represents a large hold 

The bottom of the grid is the start, and the top is the end of the route.
Here is the grid map:

{grid_str}

{"The predicted difficulty is " + difficulty if difficulty else ""}

Provide a concise but informative description of this climbing route. Don't introduce yourself and focus on being concise.
Include:
1. The overall flow/pattern of the route
2. A simple but direct, bottom to top, description of how to get from one hold to the next.

Keep your response under 200 words and focus on being practical and helpful. [/INST]
"""
    if use_local_llm:
        try:
            from llama_cpp import Llama
            from huggingface_hub import hf_hub_download
            
            try:
                model_path = hf_hub_download(
                    repo_id="TheBloke/Llama-2-7B-Chat-GGUF", 
                    filename="llama-2-7b-chat.Q4_K_M.gguf"
                )
                
                print(f"Using model from: {model_path}")
                
                # Initialize the LLM with modest parameters for Pi
                llm = Llama(
                    model_path=model_path,
                    n_ctx=2048,
                    n_threads=2
                )
                
                # Generate response
                response = llm(
                    prompt,
                    max_tokens=256,
                    temperature=0.1,
                    top_p=0.9,
                    repeat_penalty=1.2,
                    echo=False
                )
                
                return response['choices'][0]['text'].strip() or generate_generic_description(grid_map, difficulty)
                
            except Exception as e:
                print(f"Error loading or using local model: {e}")
                return generate_generic_description(grid_map, difficulty)
        
        except ImportError:
            print("Required libraries for local LLM not installed.")
            return generate_generic_description(grid_map, difficulty)
    
    elif api_url:
        try:
            headers = {
                "Authorization": f"Bearer {HF_API_TOKEN}",  # <-- Add this line
                "Content-Type": "application/json"
            }
            payload = {
                "prompt": prompt,
                "max_tokens": 300,
                "temperature": 0.7
            }
            
            response = requests.post(api_url, headers=headers, data=json.dumps(payload), timeout=10)
            if response.status_code == 200:
                return response.json().get("text", "").strip() or generate_generic_description(grid_map, difficulty)
            else:
                print(f"API error: {response.status_code} - {response.text}")
                return generate_generic_description(grid_map, difficulty)
                
        except Exception as e:
            print(f"Error using API LLM: {e}")
            return generate_generic_description(grid_map, difficulty)
    
    else:
        return generate_generic_description(grid_map, difficulty)

def generate_generic_description(grid_map: np.ndarray, difficulty: Optional[str] = None) -> str:
    """
    Generate a generic route description based on simple analysis if LLM is not available.
    
    Args:
        grid_map: 12x12 numpy array of the climbing route
        difficulty: Predicted difficulty level
        
    Returns:
        str: Generic description of the route
    """
    # Count holds
    num_small_holds = np.sum(grid_map == 1)
    num_large_holds = np.sum(grid_map == 2) // 2  # Divide by 2 since each large hold takes 2 cells
    total_holds = num_small_holds + num_large_holds
    
    # Analyze hold distribution
    left_side = np.sum(grid_map[:, :6] > 0)
    right_side = np.sum(grid_map[:, 6:] > 0)
    
    top_section = np.sum(grid_map[:4, :] > 0)
    middle_section = np.sum(grid_map[4:8, :] > 0)
    bottom_section = np.sum(grid_map[8:, :] > 0)
    
    # Generate description
    description = f"This route contains {total_holds} holds ({num_small_holds} small, {num_large_holds} large). "
    
    if left_side > right_side * 1.5:
        description += "The route favors the left side of the wall. "
    elif right_side > left_side * 1.5:
        description += "The route favors the right side of the wall. "
    else:
        description += "The route is well balanced between left and right sides. "
    
    if bottom_section > middle_section and bottom_section > top_section:
        description += "The route has more holds at the bottom, suggesting a difficult start. "
    elif top_section > middle_section and top_section > bottom_section:
        description += "The route has more holds at the top, suggesting a challenging finish. "
    elif middle_section > bottom_section and middle_section > top_section:
        description += "The crux of the route appears to be in the middle section. "
    
    if difficulty:
        description += f"The estimated difficulty is {difficulty}. "
    
    return description