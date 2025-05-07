import argparse, torch 
from pathlib import Path 
from torch.utils.data import DataLoader 
from nltk.translate.bleu_score import corpus_bleu 
from tqdm import tqdm 

from dataset import Flickr30kDataset
from model import MultiModalCaptioner

#greedy decoding 
def generate(model, pixel, tokenizer, max_len=50, device="cuda"): 
    model.eval()
    with torch.no_grad():
        B = pixel.size(0) #batch size, aka number of images in input 
        input_ids = torch.full((B, 1), tokenizer.bos_token_id, device=device, dtype=torch.long)
        
        for _ in range(max_len):
            logits, _ = model(pixel, input_ids)
            next_id = logits[:, -1].argmax(dim=-1, keepdim=True) #select argmax for next token
            input_ids = torch.cat([input_ids, next_id], dim=1) #concatenate to input
            
            if (next_id == tokenizer.eos_token_id).all(): 
                break 
        return tokenizer.batch_decode(input_ids[:, 1:], skip_special_tokens=True) #remove bos and decode

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=False, help="torch .pt checkpoint")
    args = parser.parse_args()
    
    ds = Flickr30kDataset(split="test")
    dl = DataLoader(ds, batch_size=8, shuffle=False)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MultiModalCaptioner(vocab_size=len(ds.tokenizer)).to(device)

    if args.ckpt: 
        model.load_state_dict(torch.load(args.ckpt, map_location=device))
    
    refs, hyps = [], []
    for batch in tqdm(dl, desc="Evlauating"):
        pixel = batch["pixel_values"].to(device)
        captions = generate(model, pixel, ds.tokenizer, device=device)
        hyps.extend([[c.split()] for c in captions])
        refs.extend([[ds.tokenizer.decode(batch["labels"][i], skip_special_tokens=True).split()] for i in range(len(captions))])
        
    bleu4 = corpus_bleu(refs, hyps)
    print(f"BLEU score: {bleu4:.3f}")
    return bleu4
    
    
    
if __name__ == "__main__":
    main() 
    
    
#Note to self: because of lack of eos token I would expect our inference to consistently hit max_len, lets print that to check (we could even check the average length of the captions in the dataset and compare to average of what we output)



#beam search 



#ok finish the eval script first, then go back and do the train script to add the validation set because we need to go back and add wandb, checkpoint saving, and huggingface anyway. 