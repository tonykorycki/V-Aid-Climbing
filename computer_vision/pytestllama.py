# Run this in the same Python environment where you're running your climbing code
import sys
print(sys.executable)  # Shows which Python installation you're using
try:
    import llama_cpp
    print("llama_cpp is installed correctly")
    print(f"Version: {llama_cpp.__version__}")
except ImportError as e:
    print(f"Error importing llama_cpp: {e}")