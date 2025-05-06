from pathlib import Path 
from typing import Dict 


from datasets import load_dataset 
from PIL import Image 

import torch 
from torch.utils.data import Dataset 
from transformers import ClIPProcessors, CLIPTokenizer


"""
Figure out expanding the vocabulary of our tokens 

Figure out how to deal with captions

Deal with the split (its done in the column)


"""


class Flickr30kDataset(torch.utils.data.Dataset): 
    def __init__(self, split: str = "test", root: str | Path | None = None, max_caption_length: int = 50): 
        
        self.data = load_dataset("flickr30k", cache_dir=root)[split]
        self.processor = ClIPProcessors.from_pretrained("openai/clip-vit-base-patch32")
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
        self.max_length = max_caption_length
        
    def __len__(self): 
        return len(self.data)
    
    def __getitem__(self, idx): 
        sample = self.data[idx]
        img = sample["image"]
        caption = sample["caption"]
        
        pixel = self.processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
        #tokenize caption 
        tok = self.tokenizer(caption, padding="max_length", truncation=True, max_length=self.max_length, return_tensors="pt") #(3,224,224)
        
        input_ids = tok["input_ids"].squeeze(0) #(L,)
        labels = input_ids.clone() #target output tokens that we want the decoder to generate 
        
        bos_id = self.tokenizer.bos_token_id 
        input_ids = torch.cat([torch.tensor([bos_id]), input_ids]) 
        
        # The input_ids are: [<bos>, w1, w2, ..., w_{n-1}]
        # The labels remain: [w1, w2, ..., w_{n-1}, w_n]
        
        
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
        
# if __name__ == "__main__": 
    