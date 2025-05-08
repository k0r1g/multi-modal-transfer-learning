import argparse, torch 
from pathlib import Path 
from torch.utils.data import DataLoader 
from nltk.translate.bleu_score import corpus_bleu 
from tqdm import tqdm 

from dataset import Flickr30kDataset  # your on-the-fly preprocessing dataset
from model import MultiModalCaptioner
from transformers import CLIPTokenizer


# Greedy decoding
def generate(model, pixel, tokenizer, max_len=50, device="cuda"): 
    model.eval()
    with torch.no_grad():
        B = pixel.size(0)
        input_ids = torch.full((B, 1), tokenizer.bos_token_id, device=device, dtype=torch.long)
        for _ in range(max_len):
            logits, _ = model(pixel, input_ids)
            next_id = logits[:, -1].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_id], dim=1)
            if (next_id == tokenizer.eos_token_id).all(): 
                break 
        return tokenizer.batch_decode(input_ids[:, 1:], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to .pt checkpoint")
    args = parser.parse_args()
    
    ds = Flickr30kDataset(split="test")
    dl = DataLoader(ds, batch_size=8, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    model = MultiModalCaptioner(vocab_size=tokenizer.vocab_size).to(device)

    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    refs, hyps = [], []
    print_samples = []

    for batch in tqdm(dl, desc="Evaluating"):
        pixel = batch["pixel_values"].to(device)
        captions = generate(model, pixel, tokenizer, device=device)
        hyps.extend([[c.split()] for c in captions])
        for i in range(len(captions)):
            gt = tokenizer.decode(batch["labels"][i], skip_special_tokens=True).strip()
            refs.append([gt.split()])
            if len(print_samples) < 10:
                print_samples.append((gt, captions[i]))

    bleu4 = corpus_bleu(refs, hyps)
    print(f"\nBLEU-4 score: {bleu4:.3f}\n")
    print("Sample predictions:")
    for gt, pred in print_samples:
        print(f"GT:   {gt}")
        print(f"PRED: {pred}\n")

    return bleu4


if __name__ == "__main__":
    main()


#python eval_captions.py --ckpt checkpoints/epoch11.pt
