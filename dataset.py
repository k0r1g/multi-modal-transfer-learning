from pathlib import Path 
from typing import Dict 

import torch 
from PIL import Image 
from datasets import load_dataset 
from transformers import ClIPProcessors  


class flickr30kDataset(torch.utils.data.Dataset): 
    def __init__(self, split: str = "test", max_length = 100): 
        self.dataset = load_dataset("flickr30k", split = split)
        self.processor = ClIPProcessors.from_pretrained("openai/clip-vit-base-patch32")
        self.max_length = max_length

    def __len__(self): 
        return len(self.dataset)
    
    def __getitem__(self, idx): 
        row = self.dataset[idx]
        image = row["image"]
        caption = row["caption"]
        
        proc = self.processor(
            text = [caption], 
            images = image, 
            padding = "max_length", 
            truncation = True, 
            max_length = self.max_length, 
            return_tensors = "pt"
        )
        
        #shift labels by one for LM training 
        input_ids = proc["input_ids"].squeeze(0) #shape: [max_length]
        labels = input_ids.clone()
        labels[:-1] = input_ids[1:] # Example: if input is [start, A, B, C], labels become [A, B, C, ?]
        labels[-1] = -100 #cross entropy ignores last token in label 
        
        return {
            "pixel_values": proc["pixel_values"].squeeze(0), 
            "input_ids": input_ids, 
            "labels": labels, 
        }
        
        