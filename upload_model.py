from huggingface_hub import HfApi
from pathlib import Path
import os

# Load Hugging Face token from .env or set directly here
HF_TOKEN ="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"  # replace if not using dotenv
REPO_ID = "Kogero/clip-mm-decoder"  

CHECKPOINT_PATH = Path("checkpoints/model_final_epoch10.pt")
REMOTE_FILENAME = CHECKPOINT_PATH.name
# Initialize Hugging Face API client
api = HfApi(token=HF_TOKEN)

# Create the repo if it doesn't exist (safe to re-run)
api.create_repo(repo_id=REPO_ID, exist_ok=True)

# Upload the file
api.upload_file(
    repo_id=REPO_ID,
    path_or_fileobj=str(CHECKPOINT_PATH),
    path_in_repo=REMOTE_FILENAME,
    commit_message="Upload trained model after epoch 10"
)

print(f"✅ Uploaded {REMOTE_FILENAME} to https://huggingface.co/{REPO_ID}")
