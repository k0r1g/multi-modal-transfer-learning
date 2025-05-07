import argparse 
from pathlib import Path 
import torch 
from torch.utils.data import DataLoader 
from torch.optim import AdamW
from torch.nn.functional import cross_entropy 
from tqdm import tqdm 

from dataset import Flickr30kDataset
from model import MultimodalDecoder


def parse_args(): 
    p = argsparse.ArgumentParser()
    p.add_argument("--root", type=str, default=None, help="datasets cache dir")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def main(): 
    args = parse_args()
    ds = Flickr30kDataset(split="train", root=args.root)
    dl = DataLoader(ds, batch_size = args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    model = MultiModalCaptioner(vocab_size=len(dset.tokenizer)).to(device)
    optim = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr) # freeze CLIP 
    
    if args.dry_run: 
        batch = next(iter(dl))
        pixel, inp, lbl = (t.to(device) for t in (batch["pixel_values"], batch["input_ids"], batch["labels"]))
        logits, _ = model(pixel, inp) #run forward pass 
        
             
        #note: torch.nn.functional.cross_entropy(input, target) input is (N,C) and target is (N)
        loss = cross_entropy(
            logits.view(-1, logits.size(-1)), # (B,L,V) -> (B*L, V)
            lbl.view(-1), # (B,L) -> (B*L)
            ignore_index = ds.tokenizer.pad_token_id, #note: also ignores eos token and unk token 
        )
        print("single pass OK- loss:", loss.item())
        return 

    for epoch in range(1, args.epochs + 1): 
        model.train()
        pbar = tqdm(dl, desc=f"Epoch {epoch}")
        for batch in pbar:
            pixel, inp, lbl = (t.to(device) for t in (batch["pixel_values"], batch["input_ids"], batch["labels"]))
            logits, _ = model(pixel, inp)
            loss = cross_entropy(
                logits.view(-1, logits.size(-1)), 
                lbl.view(-1), 
                ignore_index = ds.tokenizer.pad_token_id,  #note: also ignores eos token and unk token
            )
            optim.zero_grad()
            loss.backward()
            optim.step()
            pbar.set_postfix(loss={loss.item(): f"{loss.item():.3f}"})
            
    if __name__ == "__main__": 
        main()