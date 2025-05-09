import argparse, torch 
from pathlib import Path 
from torch.utils.data import DataLoader 
from nltk.translate.bleu_score import corpus_bleu 
from tqdm import tqdm 

from dataset import Flickr30kDataset  
from model import MultiModalCaptioner
from transformers import CLIPTokenizer


# Greedy decoding
def generate(model, pixel, tokenizer, max_len=50, device="cuda", 
             temperature=0.7, top_k=50, top_p=0.9, repetition_penalty=1.5, 
             num_beams=3, use_sampling=True):
    """
    Enhanced generation function with:
    - Temperature sampling
    - Top-k and top-p filtering
    - Repetition penalty
    - Optional beam search
    """
    model.eval()
    with torch.no_grad():
        B = pixel.size(0)
        input_ids = torch.full((B, 1), tokenizer.bos_token_id, device=device, dtype=torch.long)
        
        # Create text mask for the input (all 1s since there's no padding yet)
        text_mask = torch.ones((B, 1), device=device)
        
        # Track generated tokens to apply repetition penalty
        prev_tokens = [[] for _ in range(B)]
        
        for i in range(max_len):
            logits, _ = model(pixel, input_ids, text_mask)
            next_token_logits = logits[:, -1, :] / temperature  # Apply temperature
            
            # MODIFICATION: Prevent EOS from being generated in the first few tokens
            if i < 3:  # Force model to generate at least 3 tokens before EOS
                next_token_logits[:, tokenizer.eos_token_id] = -float('inf')
            
            # Apply repetition penalty
            for batch_idx in range(B):
                for prev_token in prev_tokens[batch_idx]:
                    next_token_logits[batch_idx, prev_token] /= repetition_penalty
            
            if use_sampling:
                # Top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = -float('inf')
                
                # Top-p (nucleus) filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above the threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    # Shift the indices to the right to keep also the first token above the threshold
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    
                    for batch_idx in range(B):
                        indices_to_remove = sorted_indices[batch_idx][sorted_indices_to_remove[batch_idx]]
                        next_token_logits[batch_idx, indices_to_remove] = -float('inf')
                
                # Sample from the filtered distribution
                probs = torch.softmax(next_token_logits, dim=-1)
                next_id = torch.multinomial(probs, 1)
            else:
                # Greedy decoding
                next_id = next_token_logits.argmax(dim=-1, keepdim=True)
            
            # Update prev_tokens list for repetition penalty
            for batch_idx in range(B):
                prev_tokens[batch_idx].append(next_id[batch_idx].item())
                
            # Add the predicted token to the sequence
            input_ids = torch.cat([input_ids, next_id], dim=1)
            # Update mask
            text_mask = torch.cat([text_mask, torch.ones((B, 1), device=device)], dim=1)
            
            # Early stopping if all sequences have EOS
            if (next_id == tokenizer.eos_token_id).all():
                break
                
        return tokenizer.batch_decode(input_ids[:, 1:], skip_special_tokens=True)
    
def calculate_running_bleu(refs, hyps):
    if not refs or not hyps:
        return 0 
    return corpus_bleu(refs, hyps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to .pt checkpoint")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to display during evaluation")
    parser.add_argument("--display_frequency", type=int, default=10, help="Update running metrics every N batches")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=50, help="Top-k filtering parameter")
    parser.add_argument("--top_p", type=float, default=0.9, help="Top-p filtering parameter")
    parser.add_argument("--repetition_penalty", type=float, default=1.5, help="Penalty for repeating tokens")
    parser.add_argument("--use_sampling", action="store_true", help="Use sampling instead of greedy decoding")
    parser.add_argument("--min_length", type=int, default=5, help="Minimum length of generated captions")
    args = parser.parse_args()
    
    ds = Flickr30kDataset(split="test")
    dl = DataLoader(ds, batch_size=8, shuffle=False)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    model = MultiModalCaptioner(vocab_size=tokenizer.vocab_size).to(device)

    model.load_state_dict(torch.load(args.ckpt, map_location=device))

    refs, hyps = [], []
    displayed_samples = 0
    total_samples = min(args.samples, len(ds))
    
    print(f"\n{'='*80}\nStarting evaluation with {len(ds)} test samples\n{'='*80}\n")

    for batch_idx, batch in enumerate(tqdm(dl, desc="Evaluating")):
        pixel = batch["pixel_values"].to(device)
        captions = generate(
            model, pixel, tokenizer, device=device,
            temperature=args.temperature,
            top_k=args.top_k, 
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            use_sampling=args.use_sampling
        )
        
        batch_refs = []
        batch_hyps = []
        
        #process each sample in batch 
        for i in range(len(captions)):
            # Safer: apply label mask from attention_mask
            ids = batch["labels"][i]
            mask = batch["attention_mask"][i]
            gt = tokenizer.decode(ids[mask.bool()], skip_special_tokens=True).strip()
            pred = captions[i]
            
            #add to overall metrics 
            refs.append([gt.split()])
            hyps.append(pred.split())
            
            batch_refs.append([gt.split()])
            batch_hyps.append(pred.split())
            
            if displayed_samples < total_samples: 
                print(f"\nSample {displayed_samples+1}/{total_samples}")
                print(f"Ground Truth:   {gt}")
                print(f"Prediction: {pred}\n")
                displayed_samples += 1
                
        if (batch_idx + 1) % args.display_frequency == 0 or batch_idx == len(dl) - 1: 
            running_bleu = calculate_running_bleu(refs, hyps)
            batch_bleu = calculate_running_bleu(batch_refs, batch_hyps)
            print(f"Batch {batch_idx + 1}/{len(dl)} stats:")
            print(f"Batch BLEU-4: {batch_bleu:.4f}")
            print(f"Running BLEU-4 ({len(refs)} samples): {running_bleu:.4f}")

    final_bleu = corpus_bleu(refs, hyps)
    print(f"\nFinalBLEU-4 score: {final_bleu:.3f}\n")
    print("Total samples evaluated: ", len(refs))

    return final_bleu


if __name__ == "__main__":
    main() 