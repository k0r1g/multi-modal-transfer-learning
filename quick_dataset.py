from dataset import Flickr30kDataset
from transformers import CLIPTokenizer

tokenizer=CLIPTokenizer.from_pretrained('openai/clip-vit-base-patch32')

ds=Flickr30kDataset(split='train', max_caption_length=50)
item=ds[0]
print('caption', item['input_ids'][:10])
print('labels', item['labels'][:10])
print('mask', item['attention_mask'][:10])
print('decoded input', tokenizer.decode(item['input_ids'], skip_special_tokens=False))
print('decoded target', tokenizer.decode(item['labels'], skip_special_tokens=False)) 