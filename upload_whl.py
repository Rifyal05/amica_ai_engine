import os
from huggingface_hub import HfApi
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("HF_TOKEN")
REPO_ID = os.getenv("HF_REPO_ID")

api = HfApi()

def upload_only_whl():
    dist_files = [f for f in os.listdir("./dist") if f.endswith(".whl")]
    if not dist_files:
        print("File .whl tidak ditemukan di folder dist!")
        return
    
    local_file = dist_files[0]
    cloud_name = "llama_cpp_python-0.3.16-cp312-zen4-avx512.whl"

    print(f"🚀 Uploading wheel as: {cloud_name}...")
    
    api.upload_file(
        path_or_fileobj=f"./dist/{local_file}",
        path_in_repo=cloud_name,
        repo_id=REPO_ID, # type: ignore
        token=TOKEN,
        repo_type="model"
    ) # type: ignore
    print("Berhasil! Sekarang installer Zen 4 sudah dalam 1 repository dengan modelnya.")

if __name__ == "__main__":
    upload_only_whl()