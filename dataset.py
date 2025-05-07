from pathlib import Path 
from typing import Dict, List, Tuple


from datasets import load_dataset 
from PIL import Image 

import torch 
from torch.utils.data import Dataset 
from transformers import CLIPProcessor, CLIPTokenizer


"""
We use the same vocab as CLIP including its special tokens

Figure out how to deal with captions

Deal with the split (its done in the column)

"""


class Flickr30kDataset(torch.utils.data.Dataset): 
    def __init__(self, split: str = "train", root: str | Path | None = None, max_caption_length: int = 50): 
        
        raw = load_dataset("nlphuji/flickr30k", cache_dir=root, split = "test")
        self.data = raw.filter(lambda x: x["split"] == split)
        
        #reorganise the data so one sample per image-caption pair 
        self.index: List[Tuple[int, int]] = [
            (row_id, cap_id)
            for row_id , n_caps in enumerate(self.data["caption"])
            for cap_id in range(len(n_caps)) #n_caps is always 5
        ]

        
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.max_length = max_caption_length
        
    def __len__(self): 
        return len(self.index)
    
    def __getitem__(self, idx): 
        row_id, cap_id = self.index[idx]
        sample = self.data[row_id]
        
        img = sample["image"]
        caption = sample["caption"][cap_id]
        
        pixel = self.processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
        #tokenize caption 
        tok = self.tokenizer(caption, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt", add_special_tokens=True) #(3,224,224)
        
        input_ids = tok["input_ids"].squeeze(0) # already <bos> … <eos> … <pad>
        
        #everything except the last token 
        decoder_input_ids = input_ids[:-1] 
        #everything except the first token 
        labels = input_ids[1:] 
        
        # The input_ids are: [<bos>, w1, w2, ..., w_n]
        # The labels remain: [w1, w2, ..., w_n, <eos>]
        
        pad_id = self.tokenizer.pad_token_id
        decoder_input_ids = torch.nn.functional.pad(decoder_input_ids, (0,1), value=pad_id)
        labels = torch.nn.functional.pad(labels, (0,1), value=pad_id)
    
        # The input_ids are: [<bos>, w1, w2, ..., w_n, <pad>]
        # The labels remain: [w1, w2, ..., w_n, <eos>, <pad>] note, <pad> is actually the <eos> token  for CLIP
    
        
        return {
            "pixel_values": pixel, 
            "input_ids": decoder_input_ids, 
            "labels": labels, 
        }
        
if __name__ == "__main__": 
    
    #dummy data
    dummy_image = Image.new("RGB", (224, 224), color=(255, 255, 255))
    dummy_caption = ["a cat on a mat"]

    dset = Flickr30kDataset.__new__(Flickr30kDataset)  # bypass __init__
    dset.data = [{"image": dummy_image, "caption": dummy_caption}]
    dset.index = [(0, 0)]
    dset.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=True)
    dset.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    dset.max_length = 10
    
    
    #actual tests
    sample = dset[0]
    assert sample["pixel_values"].shape == (3, 224, 224)
    
    ids = sample["input_ids"]
    lbls = sample["labels"]
    assert ids.shape == lbls.shape
    assert ids[0] == dset.tokenizer.bos_token_id #ensure it starts with BOS 
    assert lbls[0] != dset.tokenizer.bos_token_id 
    
    #PAD only at the end 
    pad_id = dset.tokenizer.pad_token_id
    pad_mask = (ids == pad_id)
    first_pad = pad_mask.float().argmax().item() #first index where pad appears 
    assert not pad_mask[:first_pad].any()



    print("Passed all tests")

    
    
    
    