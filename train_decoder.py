import argparse 
from pathlib import Path 
import time
import torch 
from torch.utils.data import DataLoader 
from torch.optim import AdamW
from torch.nn.functional import cross_entropy 
from tqdm import tqdm 

from dotenv import load_dotenv 
import wandb 
from huggingface_hub import HfApi
from dataset import Flickr30kDataset          # <-- we'll use our own Dataset class
from model import MultiModalCaptioner 
from transformers import CLIPTokenizer

import os 
import uuid 
import shutil

tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

def masked_ce_loss(logits, targets, mask):
    """
    logits: (B, L, V)
    targets: (B, L)
    mask: (B, L), 1 for real tokens, 0 for padding
    """
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),      # (B*L, V)
        targets.view(-1),                      # (B*L)
        reduction="none"
    )
    mask = mask.view(-1).float()              # (B*L)
    return (loss * mask).sum() / mask.sum()

# ------------------------------------------------------------------
# Simple collate-fn so DataLoader can stack our dicts -> batch.
# ------------------------------------------------------------------
def collate_fn(batch):
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch]) for k in keys}

def parse_args(): 
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default=None, help="datasets cache dir")
    p.add_argument("--batch_size", type=int, default=128) #prev 16
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--dry_run", action="store_true")
    p.add_argument("--project", type=str, default="clip-mm-decoder") #wandb project
    p.add_argument("--repo", type=str, default="clip-mm-decoder") #hf repo 
    p.add_argument("--save_ckpt", type=str, default=Path("checkpoints"))
    p.add_argument("--train_frac", type=float, default=1.0,
                   help="Fraction of the training split to use. Set to 0.2 to use one-fifth of the data.")
    p.add_argument("--val_frac", type=float, default=1.0,
                   help="Fraction of the validation split to use.")
    return p.parse_args()


def main(): 
    load_dotenv()
    args = parse_args()
    
    wandb_mode = "disabled" if args.dry_run else "online"
    wandb.init(project=args.project, config=vars(args), mode=wandb_mode)
    
    hf_api = HfApi(token=os.getenv("HF_TOKEN")) if not args.dry_run else None 
    run_id = uuid.uuid4().hex[:8]
    
    # ------------------------------------------------------------------
    # Use on-the-fly dataset class (no HF parquet, no streaming)
    # ------------------------------------------------------------------
    ds_train_full = Flickr30kDataset(split="train", root=args.root)
    ds_val_full = Flickr30kDataset(split="val", root=args.root)
    
    # ------------------------------------------------------------------
    # Optionally take a random subset of the dataset for faster iteration
    # ------------------------------------------------------------------
    from torch.utils.data import Subset
    import random
    if args.train_frac < 1.0:
        subset_size = int(len(ds_train_full) * args.train_frac)
        indices = random.sample(range(len(ds_train_full)), subset_size)
        ds_train = Subset(ds_train_full, indices)
    else:
        ds_train = ds_train_full

    if args.val_frac < 1.0:
        subset_size = int(len(ds_val_full) * args.val_frac)
        indices = random.sample(range(len(ds_val_full)), subset_size)
        ds_val = Subset(ds_val_full, indices)
    else:
        ds_val = ds_val_full
    
    dl_train = DataLoader(
        ds_train, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=2, 
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    dl_val = DataLoader(
        ds_val, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=2, 
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    #model and optimiser
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiModalCaptioner(vocab_size=len(tokenizer)).to(device)
    
    # Print model architecture and parameter count
    print(model)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    optim = AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr) # freeze CLIP 
    
    if args.dry_run: 
        batch = next(iter(dl_train))
        pixel, inp, lbl, mask = (
            t.to(device)
            for t in (
                batch["pixel_values"],
                batch["input_ids"],
                batch["labels"],
                batch["attention_mask"],
            )
        )
        logits, _ = model(pixel, inp, mask) #run forward pass 
        loss = masked_ce_loss(logits, lbl, mask)
        print("single pass OK- loss:", loss.item())
        return 
    
    args.save_ckpt.mkdir(parents=True, exist_ok=True)
    global_step = 0


    for epoch in range(1, args.epochs + 1): 
            
        #training loop 
        model.train()
        pbar = tqdm(dl_train, desc=f"Epoch {epoch}")
        for batch in pbar:
            start_time = time.time()
            pixel, inp, lbl, mask = (
                t.to(device)
                for t in (
                    batch["pixel_values"],
                    batch["input_ids"],
                    batch["labels"],
                    batch["attention_mask"],
                )
            )
            logits, _ = model(pixel, inp, mask)
            loss = masked_ce_loss(logits, lbl, mask)

            optim.zero_grad()
            loss.backward()
            optim.step()
            
            global_step += 1
            wandb.log({"train/loss": loss.item(), 
                      "epoch": epoch, 
                      "step": global_step})
            
            step_time = time.time() - start_time
            pbar.set_postfix(loss=f"{loss.item():.3f}", step_time=f"{step_time:.2f}s")
            
        #validation loop 
        model.eval()
        val_loss_sum, n_batches = 0.0, 0 
        with torch.no_grad():
            for batch in dl_val:
                pixel, inp, lbl, mask = (
                    t.to(device)
                    for t in (
                        batch["pixel_values"],
                        batch["input_ids"],
                        batch["labels"],
                        batch["attention_mask"],
                    )
                )
                logits, _ = model(pixel, inp, mask)
                loss = masked_ce_loss(logits, lbl, mask)
                val_loss_sum += loss.item()
                n_batches += 1
        val_loss = val_loss_sum / n_batches 
        wandb.log({"val/loss": val_loss, "epoch": epoch})
        print(f"Epoch {epoch} val loss: {val_loss:.3f}")
        
        #checkpoint at each epoch, save to wandb a
        ckpt_path = args.save_ckpt / f"epoch{epoch}.pt"
        torch.save(model.state_dict(), ckpt_path)
        wandb.save(ckpt_path)
    
    #save to hf 
    if args.repo: 
        hf_api.create_repo(args.repo, exist_ok=True)
        final_path = args.save_ckpt / f"model_final_epoch{args.epochs}.pt"
        shutil.copy(ckpt_path, final_path)
        hf_api.upload_file(
            repo_id=args.repo,
            path_or_fileobj=str(final_path),
            path_in_repo=final_path.name,
            commit_message=f"Upload model trained for {args.epochs} epochs (run {run_id})"
        )
        print(f"Model pushed to https://huggingface.co/{args.repo} - file {final_path.name}")
    wandb.finish()
            
if __name__ == "__main__": 
    main()