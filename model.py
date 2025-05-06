import math 
from typing import Tuple 

import torch 
import torch.nn as nn 
from transformers import CLIPModel, CLIPTokenizer 
from einops import rearrange, repeat 


class CLIPBackbone(nn.Module): 
    """Frozen CLIP, exposes vision and text embeddings only."""
    def __init__(self, model_name: str = "openai/clip-vit-base-patch32"): 
        super().__init__()
        self.clip = CLIPModel.from_pretrained(model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.clip.eval()
        for p in self.clip.parameters(): 
            p.requires_grad = False 
        # print("Text embedding dim:", self.clip.text_model.embeddings.token_embedding.embedding_dim)
        # print("Vision hidden dim:", self.clip.vision_model.config.hidden_size)
        # print("Text hidden dim:", self.clip.text_model.config.hidden_size)


    @torch.no_grad()
    def encode_image(self, pixel_values: torch.FloatTensor) -> torch.FloatTensor: 
        # (B, L_v, D_v)
        out = self.clip.vision_model(pixel_values).last_hidden_state
        return out
    
    @torch.no_grad()
    def embed_text(self, input_ids: torch.LongTensor) -> torch.FloatTensor: 
        # (B, L_t, D_v)
        emb = self.clip.text_model.embeddings #equivalent to nn.Embedding(...) aka embedding layer from CLIP
        out = emb(input_ids = input_ids) 
        # print("CLIP text embedding dim:", out.shape[-1])
        return out


"""
Currently using nn.MultiheadAttention for self-attention. 

Will implement in full later in the day. 

"""

class CausalSelfAttnBlock(nn.Module): 
    """A single transformer block with no cross attention"""
    def __init__(self, d_model: int, n_heads: int, dropout: float): 
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ln1 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.ln2 = nn.LayerNorm(d_model)
    
    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor):
        x = x + self.self_attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=attn_mask)[0]
        x = x + self.mlp(self.ln2(x))
        return x 

class MMDecoder(nn.Module):
    """Stack of causal self-attention blocks""" 
    def __init__(self, d_model=512, n_layers=6, n_heads=8, dropout=0.1, img_input_dim=768, txt_input_dim=512,  max_len = 77 + 50):
        super().__init__()
         # modality id embedding: 0 = image, 1 = text 
        self.type_emb = nn.Embedding(2, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.img_proj = nn.Linear(img_input_dim, d_model, bias=False)
        self.txt_proj = nn.Linear(txt_input_dim, d_model, bias=False)
        self.blocks = nn.ModuleList([CausalSelfAttnBlock(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        
    def forward(self, h_img: torch.Tensor, h_txt: torch.Tensor, txt_input_ids: torch.Tensor):
        """
        h_img : (B, L_v, D_v)  – visiion token embeddings 
        h_txt : (B, L_t, D_v)  – token embeddings  (shifted‑right captions)
        txt_input_ids : (B, L_t) – for position indexing
        """
        B, L_v, _ = h_img.shape
        _, L_t, _ = h_txt.shape
        assert L_t == txt_input_ids.shape[1]
        
        # project to decoder dimension 
        h_img = self.img_proj(h_img)
        h_txt = self.txt_proj(h_txt)
        
        # add position embeddings and type embeddings 
        img_pos_ids = torch.arange(L_v, device=h_img.device)
        txt_pos_ids = torch.arange(L_t, device=h_txt.device)
        h_img = h_img + self.type_emb.weight[0] + self.pos_emb(img_pos_ids) # h_img: shape (B, L_v, d_model) — e.g. (B, 77, 512)
        h_txt = h_txt + self.type_emb.weight[1] + self.pos_emb(txt_pos_ids) # h_txt: shape (B, L_t, d_model) — e.g. (B, 50, 512)
        
        x = torch.cat([h_img, h_txt], dim=1) #concatenate across the sequence axis, output shape (B, L_total, d_model) — e.g. (B, 127, 512)
        L_total = L_v + L_t 
        
        # causal mask so token *i* attends to <= i  (images patches see all)
        causal_mask = torch.full((L_total, L_total), float("-inf"), device=x.device)
        causal_mask[:, :L_v] = 0 
        causal_mask[L_v:, L_v:] = torch.tril(torch.zeros(L_t, L_t))
        
        for blk in self.blocks: 
            x = blk(x, causal_mask)
        
        return self.ln_f(x) # (B, L_total, D)


# The overall purpose: Given an image and a prefix caption, predict the next caption tokens one at a time, using a decoder-only architecture.
## This is a decoder-only model, so captions are passed in already shifted-right during training. That is: the input sequence is like <BOS> A dog is and the model learns to predict A, then dog, then is, etc.

class MultiModalCaptioner(nn.Module): 
    """
    Full model = frozen CLIP + decoder + projection head 
    """
    def __init__(self, vocab_size: int, d_model: int = 512, n_layers: int = 6, n_heads: int = 8, dropout: float = 0.1): 
        super().__init__()
        self.backbone = CLIPBackbone()
        self.decoder = MMDecoder(d_model = d_model, n_layers = n_layers, n_heads = n_heads, dropout = dropout)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, pixel_values: torch.Tensor, caption_input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        caption_input_ids – already shifted‑right (<BOS> w_t-1 ...)
        returns logits over vocab & concat hidden states (for optional use).
        """
        
        h_img = self.backbone.encode_image(pixel_values) #(B, L_v, 768)
        #must give it tokenised captions 
        h_txt = self.backbone.embed_text(caption_input_ids) #(B, L_t, 768)

        # pass through decoder 
        h_dec = self.decoder(h_img, h_txt, caption_input_ids) #(B, L_total, d_model)

        # project to vocab size 
        _ , L_v, _ = h_img.shape
        h_t_only = h_dec[:, L_v:, :] #(B, L_t, vocab_size) aka drop vision part 
        logits = self.lm_head(h_t_only) 
        
        return logits, h_dec 
    
  
#---- Testing ------------------------------
if __name__ == "__main__": 
    B, L_img, L_txt = 2, 50, 10 
    D_txt_clip, D_img_clip, D_dec = 512, 768, 512 
    
    #---test CLIPBackbone-----
    backbone = CLIPBackbone()
    dummy_img = torch.randn(B, 3, 224, 224) # b, 3 channels, 224x224 pixels 
    img_h = backbone.encode_image(dummy_img)
    assert img_h.shape == torch.Size([B, L_img, D_img_clip])
    
    txt_ids =  torch.randint(0, backbone.tokenizer.vocab_size, (B, L_txt))
    # print("txt_ids:", txt_ids)
    txt_h = backbone.embed_text(txt_ids)
    
    # print("txt_h.shape:", txt_h.shape)
    # print("txt_ids.shape:", txt_ids.shape)
    # print("backbone.tokenizer.vocab_size:", backbone.tokenizer.vocab_size)

    assert txt_h.shape == torch.Size([B, L_txt, D_txt_clip]) # 2, 10, 768
    
    #---test CausalSelfAttnBlock-----
    block = CausalSelfAttnBlock(d_model = D_dec, n_heads = 8, dropout = 0.1)
    dummy_x = torch.randn(B, L_img + L_txt, D_dec)
    full_mask = torch.zeros(L_img + L_txt, L_img + L_txt)
    y = block(dummy_x, full_mask)
    assert y.shape == dummy_x.shape
    
    #---test MMDecoder-----
    decoder = MMDecoder(d_model = D_dec, n_layers=2, n_heads=8,  img_input_dim=D_img_clip, txt_input_dim=D_txt_clip, max_len=L_img + L_txt)
    dec_out = decoder(img_h, txt_h, txt_ids)
    assert dec_out.shape == torch.Size([B, L_img + L_txt, D_dec])
    
    #---test MultiModalCaptioner-----
    captioner = MultiModalCaptioner(vocab_size=backbone.tokenizer.vocab_size)
    logits, hs = captioner(dummy_img, txt_ids)
    assert logits.shape == torch.Size([B, L_txt, backbone.tokenizer.vocab_size])
    assert hs.shape == torch.Size([B, L_img + L_txt, D_dec])

    print("All tests passed!")