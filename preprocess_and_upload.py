# preprocess_and_upload.py  (FAST, multi-process)
import os
from pathlib import Path
from functools import partial
from datasets import load_dataset, DatasetDict
from transformers import CLIPProcessor, CLIPTokenizer
from huggingface_hub import HfApi
from dotenv import load_dotenv
import torch

# ------------------- config -------------------
DATASET_NAME = "nlphuji/flickr30k"
REPO_ID      = "k0r1g/flickr30k-clip-preprocessed"
CACHE_DIR    = "hf_cache"
MAX_LEN      = 50
NUM_PROC     = os.cpu_count() // 2 or 2   # use half your CPUs
# ---------------------------------------------

load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
assert HF_TOKEN, "HF_TOKEN missing from .env"
api = HfApi(token=HF_TOKEN)

tokenizer  = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
processor  = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
PAD_ID     = tokenizer.pad_token_id


def expand_and_preprocess(batch, split_tag):
    """
    Receives a *batch* where each element has:
        image  (PIL.Image)
        caption (list[str] length=5)
    Returns a dict of flattened / preprocessed arrays.
    """
    images   = []
    captions = []

    # 1) expand 5 captions -> 5 samples per image
    for img, caps in zip(batch["image"], batch["caption"]):
        images.extend([img] * len(caps))
        captions.extend(caps)

    # 2) image processing (vectorised)
    pixel_tensors = processor(images=images,
                              return_tensors="pt")["pixel_values"]        # (N,3,224,224)

    # 3) tokenisation (vectorised)
    ids = tokenizer(captions,
                    padding="max_length",
                    truncation=True,
                    max_length=MAX_LEN,
                    return_tensors="pt",
                    add_special_tokens=True)["input_ids"]                # (N, MAX_LEN)

    # 4) build decoder inputs / labels
    dec_in  = torch.nn.functional.pad(ids[:, :-1], (0, 1), value=PAD_ID)
    labels  = torch.nn.functional.pad(ids[:, 1:],  (0, 1), value=PAD_ID)

    return {
        "pixel_values": pixel_tensors.numpy(),   # keep as float32
        "input_ids":    dec_in.tolist(),
        "labels":       labels.tolist(),
    }


def build_split(split_tag: str):
    """Replicates your custom split filtering & returns pre-processed Dataset."""
    raw = load_dataset(DATASET_NAME,
                       split="test",
                       cache_dir=CACHE_DIR)

    raw = raw.filter(lambda x: x["split"] == split_tag)

    # batched map → fast & parallel
    ds = raw.map(
        partial(expand_and_preprocess, split_tag=split_tag),
        batched=True,
        batch_size=128,
        num_proc=NUM_PROC,
        remove_columns=raw.column_names,
        desc=f"Building '{split_tag}'",
    )
    return ds


def main():
    os.makedirs(CACHE_DIR, exist_ok=True)

    ds_train = build_split("train")
    ds_val   = build_split("val")
    ds_test  = build_split("test")

    ds_dict = DatasetDict({"train": ds_train,
                           "val":   ds_val,
                           "test":  ds_test})

    print("Uploading to hub …")
    ds_dict.push_to_hub(
        REPO_ID,
        token=HF_TOKEN,
        private=False,
        max_shard_size="2GB",
    )
    print(f"Uploaded → https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
