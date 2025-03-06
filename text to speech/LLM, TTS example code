from llama_cpp import Llama
import pyttsx3
import json

# Load Local Model (Modify path based on your setup)
llm = Llama(model_path="./mistral-7b.Q5_K_M.gguf")  # Adjust for your model

# Example climbing grid
climbing_grid = [
    {"type": "foothold", "position": (1, 1)},
    {"type": "crimp", "position": (2, 4)},
    {"type": "sloper", "position": (5, 2)},
    {"type": "jug", "position": (6, 4)},
    {"type": "foothold", "position": (3, 2)}
]

climbing_data = json.dumps(climbing_grid)

prompt = f"""
Given the following climbing holds:

{climbing_data}

Generate a short climbing route description (under 25 seconds) for a visually impaired climber.
"""

# Generate response
response = llm(prompt)["choices"][0]["text"]
print("Generated Route:", response)

# Convert to speech
engine = pyttsx3.init()
engine.say(response)
engine.runAndWait()
