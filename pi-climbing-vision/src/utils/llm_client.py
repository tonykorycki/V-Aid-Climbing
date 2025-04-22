import requests
import json
import numpy as np
from typing import Dict, Optional
from paths import HF_API_TOKEN  # <-- Add this import

def generate_route_description(grid_map: np.ndarray, 
                               difficulty: Optional[str] = None, 
                               use_local_llm: bool = False, 
                               api_url: Optional[str] = None) -> str:
    # Create a clearer ASCII representation of the grid (similar to display_text_grid)
    grid_str = ""
    
    # Add column markers (double digit spacing)
    grid_str += "  " + "".join([f"{i:2d}" for i in range(grid_map.shape[1])]) + "\n"
    grid_str += "  " + "-" * (grid_map.shape[1] * 2) + "\n"
    
    # Add rows without flipping
    for i, row in enumerate(grid_map):
        line = f"{i:2d}|"
        for cell in row:
            if cell == 0:
                line += ". "  # Empty space
            elif cell == 1:
                line += "S "  # Small hold
            elif cell == 2:
                line += "L "  # Large hold
        grid_str += line + "\n"
    
    # Add legend
    grid_str += "\nLegend: '.' = Empty, 'S' = Small hold, 'L' = Large hold\n"
    
    # Create the prompt for the LLM
    prompt = f"""[INST]
You are analyzing a climbing route represented in a grid.

Legend:
- . (dot) = empty space (no hold)
- S = small hold
- L = large hold/volume

The grid is oriented as follows:
- Top row (row 0) is the finish of the climb
- Bottom row (row 11) is the start
- Columns 0-5 represent the left side of the wall
- Columns 6-11 represent the right side of the wall

Here is the climbing route grid:

{grid_str}

{"The estimated difficulty is " + difficulty if difficulty else ""}

Describe this climbing route for a visually impaired climber. Be concise but informative (150-200 words).
Focus on:
1. The overall path and flow from bottom to top
2. Location of key holds (left side vs right side)
3. Transitions between sections of the route
4. Any particularly challenging sequences

Avoid using grid coordinates or technical jargon. Describe the route in practical terms.
Make it easy to understand for someone who cannot see the grid.
[/INST]
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
                    temperature=0.2,
                    top_p=0.95,
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
        # Prepare the API request
        headers = {
            "Authorization": f"Bearer {HF_API_TOKEN}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 300,
                "temperature": 0.1,
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