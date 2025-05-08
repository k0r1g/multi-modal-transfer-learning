# preprocess_and_upload.py
import os, math, json, shutil
from pathlib import Path
from tqdm import tqdm

import torch
from datasets import load_dataset, Dataset, DatasetDict

from transformers import CLIPProcessor, CLIPTokenizer
from huggingface_hub import HfApi
from dotenv import load_dotenv


DATASET_NAME   = "nlphuji/flickr30k"          # upstream dataset
REPO_ID        = "k0r1g/flickr30k-clip-preprocessed"  # HF repo to create / update
MAX_LEN        = 50                           # caption length used in training
CACHE_DIR      = "hf_cache"                  # local tmp folder for .arrow files


def main() -> None:
    load_dotenv()
    hf_token = os.getenv("HF_TOKEN")
    assert hf_token, "⚠️  HF_TOKEN not found in .env"
    api = HfApi(token=hf_token)

    # Shared tokenizer / processor (loaded once)
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def preprocess_split(split_tag: str) -> Dataset:
        """
        Build a Dataset that exactly matches Flickr30kDataset(split_tag)
        but fully pre-processed.
        """
        raw = load_dataset(
            DATASET_NAME,
            split="test",
            cache_dir=CACHE_DIR,
        )
        # Sub-select rows whose "split" column matches train / val / test
        raw = raw.filter(lambda x: x["split"] == split_tag)

        rows = []
        for sample in tqdm(raw, desc=f"⇢ {split_tag}", unit="img"):
            image = sample["image"]
            for caption in sample["caption"]:       # 5 captions per image
                # 1) image → float tensor (3,224,224)
                pixel = processor(images=image, return_tensors="pt")[
                    "pixel_values"
                ].squeeze(0)            # (3,224,224)

                # 2) caption → token IDs (BOS ... EOS PAD)
                ids = tokenizer(
                    caption,
                    padding="max_length",
                    truncation=True,
                    max_length=MAX_LEN,
                    return_tensors="pt",
                    add_special_tokens=True,
                )["input_ids"].squeeze(0)            # (MAX_LEN,)

                # 3) Build decoder inputs / labels
                dec_in   = torch.nn.functional.pad(ids[:-1], (0, 1), value=tokenizer.pad_token_id)
                labels   = torch.nn.functional.pad(ids[1:],  (0, 1), value=tokenizer.pad_token_id)

                rows.append(
                    {
                        "pixel_values":  pixel.numpy(),      # keep as float32
                        "input_ids":     dec_in.tolist(),
                        "labels":        labels.tolist(),
                    }
                )

        return Dataset.from_list(rows)


    os.makedirs(CACHE_DIR, exist_ok=True)
    print("Pre-processing Flickr30k …")
    ds_train = preprocess_split("train")
    ds_val   = preprocess_split("val")
    ds_test  = preprocess_split("test")


    ds_dict = DatasetDict({"train": ds_train, "val": ds_val, "test": ds_test})
    print("\n Uploading to the Hugging Face Hub …")
    ds_dict.push_to_hub(
        REPO_ID,
        token=hf_token,
        private=False,             # set True if you want it private
        max_shard_size="2GB",      # splits files if >2 GB
    )
    print(f" Uploaded → https://huggingface.co/datasets/{REPO_ID}")

if __name__ == "__main__":
    main()
