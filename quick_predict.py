import torch
from transformers import CLIPTokenizer
from dataset import Flickr30kDataset
from torch.utils.data import DataLoader
from model import MultiModalCaptioner

device='cpu'

tokenizer=CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
model=MultiModalCaptioner(vocab_size=tokenizer.vocab_size)
model.load_state_dict(torch.load('checkpoints/model_final_epoch10.pt', map_location=device))
model.eval()

ds=Flickr30kDataset(split='val', max_caption_length=50)
first=ds[0]
pixel=first['pixel_values'].unsqueeze(0)
bos = torch.full((1,1), tokenizer.bos_token_id)
mask = torch.ones((1,1))
with torch.no_grad():
    logits,_ = model(pixel, bos, mask)
    next_token_id=logits[:, -1, :].argmax(-1).item()
print('Pred token', tokenizer.decode([next_token_id])) 