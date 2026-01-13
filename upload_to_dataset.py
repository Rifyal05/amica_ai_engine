import os
from huggingface_hub import HfApi
from dotenv import load_dotenv

load_dotenv()

TOKEN = os.getenv("HF_TOKEN")
DATASET_ID = os.getenv("HF_DATASET_ID")

api = HfApi()

def upload_to_dataset():
    try:
        api.create_repo(
            repo_id=DATASET_ID,  # type: ignore
            token=TOKEN, 
            repo_type="dataset", 
            private=False 
        )
        print(f"Repo Dataset {DATASET_ID} berhasil dibuat!")
    except:
        print(f"Repo {DATASET_ID} sudah ada, lanjut upload...")

    dist_files = [f for f in os.listdir("./dist") if f.endswith(".whl")]
    if not dist_files:
        print("File .whl tidak ditemukan di folder dist!")
        return
    
    local_file = dist_files[0]
    cloud_name = "llama_cpp_python-0.3.16-cp312-cp312-manylinux_2_39_x86_64.whl"

    print(f"🚀 Mengirim 'Mesin Turbo' Zen 4 ke Dataset: {cloud_name}...")
    
    api.upload_file(
        path_or_fileobj=f"./dist/{local_file}",
        path_in_repo=cloud_name,
        repo_id=DATASET_ID, # type: ignore
        token=TOKEN,
        repo_type="dataset"
    ) # type: ignore
    print("\n✅ SELESAI! Sekarang semua orang bisa melihat hasil karyamu di profil.")

if __name__ == "__main__":
    upload_to_dataset()