import os
from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()
HF = os.getenv("HF")
api = HfApi()

api.upload_folder(
    folder_path="data/aitdna/aitdna/session_0",
    repo_id="marinajim/AITDNA",
    repo_type="dataset",
    path_in_repo="aitdna/",
    token=HF
)