import argparse 
from pathlib import Path 
import torch 
from torch.utils.data import DataLoader 
from torch.optim import AdamW
from torch.nn.functional import cross_entropy 
from tqdm import tqdm 

from dotenv import load_dotenv 
import wandb 
from huggingface_hub import HfApi
from dataset import Flickr30kDataset
from model import MultiModalCaptioner 


import os 
import uuid 
import shutil


def parse_args(): 
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=None, help="datasets cache dir")
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--project", type=str, default="clip-mm-decoder") #wandb project
    p.add_argument("--repo", type=str, default="clip-mm-decoder") #hf repo 
    p.add_argument("--save_ckpt", type=str, default=Path("checkpoints"))
    return p.parse_args()


def main(): 
    load_dotenv()
    args = parse_args()
    
    wandb_mode = "disabled" if args.dry_run else "online"
    wandb.init(project=args.project, config=vars(args), mode=wandb_mode)
    
    hf_api = HfApi(token=os.getenv("HF_TOKEN")) if not args.dry_run else None 
    run_id = uuid.uuid4().hex[:8]
    
    #training set 
    ds_train = Flickr30kDataset(split="train", root=args.root)
    dl_train = DataLoader(ds_train, batch_size = args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    
    #validation set 
    ds_val= Flickr30kDataset(split="val", root=args.root)
    dl_val = DataLoader(ds_val, batch_size = args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    #model and optimiser
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiModalCaptioner(vocab_size=len(ds_train.tokenizer)).to(device)
    
    # Print model architecture and parameter count
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    optim = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr) # freeze CLIP 
    
    if args.dry_run: 
        batch = next(iter(dl_train))
        pixel, inp, lbl = (t.to(device) for t in (batch["pixel_values"], batch["input_ids"], batch["labels"]))
        logits, _ = model(pixel, inp) #run forward pass 
        
             
        #note: torch.nn.functional.cross_entropy(input, target) input is (N,C) and target is (N)
        loss = cross_entropy(
            logits.view(-1, logits.size(-1)), # (B,L,V) -> (B*L, V)
            lbl.view(-1), # (B,L) -> (B*L)
            ignore_index = ds_train.tokenizer.pad_token_id, #note: also ignores eos token and unk token 
        )
        print("single pass OK- loss:", loss.item())
        return 
    
    args.save_dir.mkdir(parents=True, exist_ok=True)
    global_step = 0


    for epoch in range(1, args.epochs + 1): 
        
        #training loop 
        model.train()
        pbar = tqdm(dl_train, desc=f"Epoch {epoch}")
        for batch in pbar:
            pixel, inp, lbl = (t.to(device) for t in (batch["pixel_values"], batch["input_ids"], batch["labels"]))
            logits, _ = model(pixel, inp)
            loss = cross_entropy(
                logits.view(-1, logits.size(-1)), 
                lbl.view(-1), 
                ignore_index = ds_train.tokenizer.pad_token_id,  #note: also ignores eos token and unk token
            )
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            global_step += 1
            wandb.log({"train/loss": loss.item(), 
                      "epoch": epoch, 
                      "step": global_step})
            pbar.set_postfix(loss=f"{loss.item():.3f}")
            
        #validation loop 
        model.eval()
        val_loss_sum, n_val_tokens = 0.0, 0 
        with torch.no_grad():
            for batch in dl_val:
                pixel, inp, lbl = (t.to(device) for t in (batch["pixel_values"], batch["input_ids"], batch["labels"]))
                logits, _ = model(pixel, inp)
                loss = cross_entropy(
                    logits.view(-1, logits.size(-1)), 
                    lbl.view(-1), 
                    ignore_index = ds_val.tokenizer.pad_token_id, 
                )
                val_loss_sum += loss.item()
                n_val_tokens += (lbl != ds_val.tokenizer.pad_token_id).sum().item()
        val_loss = val_loss_sum / n_val_tokens 
        wandb.log({"val/loss": val_loss, "epoch": epoch})
        print(f"Epoch {epoch} val loss: {val_loss:.3f}")
        
        #checkpoint at each epoch, save to wandb a
        ckpt_path = args.save_dir / f"epoch{epoch}.pt"
        torch.save(model.state_dict(), ckpt_path)
        wandb.save(ckpt_path)
    
    #save to hf 
    if args.repo: 
        hf_api.create_repo(args.repo, exist_ok=True)
        final_path = args.save_dir / f"model_final_epoch{args.epoch}.pt"
        shutil.copy(ckpt_path, final_path)
        hf_api.upload_file(
            repo_id=args.repo,
            path_or_fileobj=str(final_path),
            path_in_repo=final_path.name,
            commit_message=f"Upload model trained for {args.epoch} epochs (run {run_id})"
        )
        print(f"Model pushed to https://huggingface.co/{args.repo} - file {final_path.name}")
    wandb.finish()
            
if __name__ == "__main__": 
    main()