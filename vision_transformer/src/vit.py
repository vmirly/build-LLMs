import torch

from torch import nn
from torch.nn import functional as F

# signature from HF:
# ViTSdpaSelfAttention(
#  (query): Linear(in_features=768, out_features=768, bias=True)
#  (key): Linear(in_features=768, out_features=768, bias=True)
#  (value): Linear(in_features=768, out_features=768, bias=True)
#  (dropout): Dropout(p=0.0, inplace=False)
#)
# ViTSelfOutput(
#  (dense): Linear(in_features=768, out_features=768, bias=True)
#  (dropout): Dropout(p=0.0, inplace=False)
#)

class MultiheadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.num_heads = cfg.num_heads
        self.p_drop = cfg.p_drop
        self.scale = cfg.embed_dim ** -0.5

        self.norm = nn.LayerNorm(cfg.embed_dim)
        self.resid_dropout = nn.Dropout(cfg.p_drop)
        self.c_attn = nn.Linear(cfg.embed_dim, cfg.embed_dim * 3)
        self.out_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim)

    def forward(self, x):
        qkv = self.c_attn(x)
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # q,k,v: [B, T, C]

        # Split the queries, keys, and values into multiple heads
        B, T, C = q.size()
        q = q.view(B, T, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.view(B, T, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.view(B, T, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # q,k,v: [B, nh, T, C//nh]

        # Flash Attention
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.p_drop
        )
        out = self.attn_dropout(out)

        # Concatenate the heads and apply a linear transformation
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        # out: [B, T, C]
        out = self.resid_dropout(out)
        return out
