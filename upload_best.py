from huggingface_hub import HfApi
import shutil
from pathlib import Path
import os

# ---- CONFIG ----
HF_TOKEN = ""  # replace this if needed
REPO_ID = "Kogero/clip-mm-decoder"
BEST_EPOCH = 11
CKPT_DIR = Path("checkpoints")  # or wherever your ckpts are saved

# ---- INIT ----
api = HfApi(token=HF_TOKEN)
best_ckpt = CKPT_DIR / f"epoch{BEST_EPOCH}.pt"
upload_name = f"best_model_epoch{BEST_EPOCH}.pt"
upload_path = CKPT_DIR / upload_name

# ---- COPY AND UPLOAD ----
shutil.copy(best_ckpt, upload_path)

api.upload_file(
    repo_id=REPO_ID,
    path_or_fileobj=str(upload_path),
    path_in_repo=upload_name,
    commit_message=f"Uploading best checkpoint (epoch {BEST_EPOCH})"
)

print(f"✅ Uploaded: {upload_name} to https://huggingface.co/{REPO_ID}/blob/main/{upload_name}")
