import os
import subprocess
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")
WHL_REPO = "cloverxion/llama_cpp_python-0.3.16-cp312-cp312-manylinux_2_39_x86_64"
WHL_FILE = "llama_cpp_python-0.3.16-cp312-cp312-manylinux_2_39_x86_64.whl"
MODEL_REPO = "cloverxion/gemma-3-1b-it-q4_km.gguf"
MODEL_FILE = "gemma-3-1b-it-q4_km.gguf"

def run_command(cmd):
    subprocess.run(cmd, shell=True, check=True)

def setup():
    os.makedirs("models", exist_ok=True)
    os.makedirs("chroma_db", exist_ok=True)

    print(" Step 1: Installing Standard Requirements...")
    run_command("pip install -r requirements.txt")

    print("Step 2: Downloading Zen 4 Optimized Wheel...")
    whl_path = hf_hub_download(
        repo_id=WHL_REPO,
        filename=WHL_FILE,
        repo_type="dataset",
        token=HF_TOKEN
    )

    print("Step 3: Installing Optimized llama-cpp-python...")
    run_command(f"pip install {whl_path} --force-reinstall")

    print("Step 4: Fetching Gemma 3 1B Model...")
    model_path = hf_hub_download(
        repo_id=MODEL_REPO,
        filename=MODEL_FILE,
        token=HF_TOKEN,
        local_dir="models",
        local_dir_use_symlinks=False
    )

    print("\nSYSTEM READY!")
    print("Run with: python main.py")

if __name__ == "__main__":
    setup()