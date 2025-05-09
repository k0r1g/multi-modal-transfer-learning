import torch
from transformers import CLIPTokenizer
from model import MultiModalCaptioner
from PIL import Image
import numpy as np
import requests
from io import BytesIO
import torch.nn.functional as F

# Load model and tokenizer
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
model = MultiModalCaptioner(vocab_size=tokenizer.vocab_size).to(device)
model.load_state_dict(torch.load('checkpoints/model_final_epoch10.pt', map_location=device))
model.eval()

print("Model loaded successfully!")
print(f"BOS token id: {tokenizer.bos_token_id}")
print(f"EOS token id: {tokenizer.eos_token_id}")

# Either create a dummy image or load a test image
# For testing with random image:
pixel_values = torch.randn(1, 3, 224, 224).to(device)

print("\n==== Debugging Generation Process ====")

# Initial token setup
input_ids = torch.full((1, 1), tokenizer.bos_token_id, device=device, dtype=torch.long)
text_mask = torch.ones((1, 1), device=device)

print(f"Initial input_ids: {input_ids}")
print(f"Initial text_mask: {text_mask}")

# Generation loop
print("\n---- Generation Steps ----")
for i in range(10):  # Generate up to 10 tokens for debugging
    with torch.no_grad():
        # Get model output
        logits, _ = model(pixel_values, input_ids, text_mask)
        next_token_logits = logits[:, -1, :]
        
        # Print top 5 tokens and their probabilities
        probs = F.softmax(next_token_logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, 5, dim=-1)
        
        print(f"\nStep {i+1}:")
        print(f"Current sequence: {tokenizer.decode(input_ids[0], skip_special_tokens=False)}")
        print("Top 5 likely next tokens:")
        for j in range(5):
            token_id = top_indices[0, j].item()
            token_prob = top_probs[0, j].item()
            token_text = tokenizer.decode([token_id])
            print(f"  {token_text} (ID: {token_id}, Prob: {token_prob:.4f})")
        
        # Get the next token (greedy for debugging)
        next_id = next_token_logits.argmax(dim=-1, keepdim=True)
        print(f"Selected token: {tokenizer.decode([next_id.item()])}")
        
        # Check if EOS token is generated
        if next_id.item() == tokenizer.eos_token_id:
            print("EOS token generated, stopping.")
            break
            
        # Add the predicted token to the sequence
        input_ids = torch.cat([input_ids, next_id], dim=1)
        text_mask = torch.cat([text_mask, torch.ones((1, 1), device=device)], dim=1)
        
        # Print current sequence
        print(f"Sequence after step {i+1}: {tokenizer.decode(input_ids[0], skip_special_tokens=False)}")

print("\n==== Final Output ====")
print(f"Generated text: {tokenizer.decode(input_ids[0], skip_special_tokens=True)}")
print(f"Raw sequence: {input_ids[0].tolist()}") 