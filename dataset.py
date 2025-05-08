from pathlib import Path
from typing import List, Tuple

from datasets import load_dataset
from PIL import Image

import torch
from torch.utils.data import Dataset
from transformers import CLIPProcessor, CLIPTokenizer

"""
We use the same vocab as CLIP including its special tokens.

Flickr30k quirk:
  • Hugging Face repo `nlphuji/flickr30k` exposes ONE split ("test")
  • Real split name is stored in column `split` with values {"train", "val", "test"}
This Dataset filters by that column.
"""


class Flickr30kDataset(Dataset):
    def __init__(
        self,
        split: str = "train",
        root: str | Path | None = None,
        max_caption_length: int = 50,
    ):
        # Load single HF "test" shard only once
        raw = load_dataset("nlphuji/flickr30k", cache_dir=root, split="test")

        # Filter by true split column
        self.data = raw.filter(lambda x: x["split"] == split)

        # one entry per (image, caption) pair
        self.index: List[Tuple[int, int]] = [
            (row_id, cap_id)
            for row_id, caps in enumerate(self.data["caption"])
            for cap_id in range(len(caps))  # always 5 captions
        ]

        self.processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32", use_fast=True
        )
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.max_length = max_caption_length

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        row_id, cap_id = self.index[idx]
        sample = self.data[row_id]

        img: Image.Image = sample["image"]
        caption: str = sample["caption"][cap_id]

        pixel = self.processor(images=img, return_tensors="pt")[
            "pixel_values"
        ].squeeze(0)

        tok = self.tokenizer(
            caption,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = tok["input_ids"].squeeze(0)
        attention_mask = tok["attention_mask"].squeeze(0)

        pad_id = self.tokenizer.pad_token_id
        decoder_input_ids = torch.nn.functional.pad(input_ids[:-1], (0, 1), value=pad_id)
        labels = torch.nn.functional.pad(input_ids[1:], (0, 1), value=pad_id)

        return {
            "pixel_values": pixel,
            "input_ids": decoder_input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }


# ----------------- sanity check -----------------
if __name__ == "__main__":
    ds = Flickr30kDataset(split="train", max_caption_length=16)
    s = ds[0]
    assert s["pixel_values"].shape == (3, 224, 224)
    assert s["input_ids"].shape == s["labels"].shape
    print("Dataset OK")
