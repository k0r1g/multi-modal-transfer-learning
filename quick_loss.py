import torch
from transformers import CLIPTokenizer
from dataset import Flickr30kDataset
from torch.utils.data import DataLoader
from model import MultiModalCaptioner
from train_decoder import masked_ce_loss as mce

tokenizer = CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')
model = MultiModalCaptioner(vocab_size=tokenizer.vocab_size)
model.load_state_dict(torch.load('checkpoints/model_final_epoch10.pt', map_location='cpu'))
model.eval()

ds = Flickr30kDataset(split='val', max_caption_length=50)
dl = DataLoader(ds, batch_size=4, shuffle=False)

batch = next(iter(dl))
pixel = batch['pixel_values']
input_ids = batch['input_ids']
labels = batch['labels']
mask = batch['attention_mask']

with torch.no_grad():
    logits, _ = model(pixel, input_ids, mask)
    loss = mce(logits, labels, mask)
print('loss', loss.item()) 