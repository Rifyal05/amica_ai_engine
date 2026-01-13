import os
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv

load_dotenv()

repo_id = "cloverxion/gemma-3-1b-it-q4_km.gguf"
filename = "gemma-3-1b-it-q4_km.gguf"
token = os.getenv("HF_TOKEN")

print(f"Sedang mengunduh model {filename}...")
path = hf_hub_download(
    repo_id=repo_id, 
    filename=filename, 
    token=token,
    local_dir="models",
    local_dir_use_symlinks=False
)
print(f"Model sudah sampai di: {path}")