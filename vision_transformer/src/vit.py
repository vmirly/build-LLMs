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

class MSA(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.num_heads = cfg.num_heads
        self.p_drop = cfg.p_drop
        self.scale = cfg.embed_dim ** -0.5

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

        # Concatenate the heads and apply a linear transformation
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        out = self.out_proj(out)
        # out: [B, T, C]
        out = self.resid_dropout(out)
        return out


# signature from HF:
# ViTIntermediate(
#  (dense): Linear(in_features=768, out_features=3072, bias=True)
#  (intermediate_act_fn): GELUActivation()
#)
# ViTOutput(
#  (dense): Linear(in_features=3072, out_features=768, bias=True)
#  (dropout): Dropout(p=0.0, inplace=False)
#)
class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.fc1 = nn.Linear(cfg.embed_dim, cfg.embed_dim * 4)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(cfg.embed_dim * 4, cfg.embed_dim)
        self.resid_dropout = nn.Dropout(cfg.p_drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.resid_dropout(x)
        return x
    
# signature from HF:
# ViTLayer(
#  (attention): ViTSdpaSelfAttention(...)
#  (intermediate): ViTIntermediate(...)
#  (output): ViTOutput(...)
#  (layernorm_before): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
#  (layernorm_after): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
#)
class ViTLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.attn = MSA(cfg)
        self.ffn = MLP(cfg)
        self.norm1 = nn.LayerNorm(cfg.embed_dim)
        self.norm2 = nn.LayerNorm(cfg.embed_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
    
# signature from HF:
# ViTEncoder(
#  (layer): ModuleList(
#    (0-11): ViTLayer(
#      (attention): ViTSdpaSelfAttention(...)
#      (intermediate): ViTIntermediate(...)
#      (output): ViTOutput(...)
#      (layernorm_before): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
#      (layernorm_after): LayerNorm((768,), eps=1e-12, elementwise_affine=True)
#    )
#  )
#)
class ViTEncoder(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        self.layers = nn.ModuleList([
            ViTLayer(cfg) for _ in range(cfg.num_layers)
        ])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

# vit embedding:
# signature from HF
# ViTEmbeddings(
#  (patch_embeddings): PatchEmbeddings(
#    (projection): Conv2d(3, 768, kernel_size=(16, 16), stride=(16, 16))
#  )
#  (dropout): Dropout(p=0.0, inplace=False)
#)
class ViTEmbeddings(nn.Module):
    def __init__(self, cfg):
        super().__init__()

        seq_len = 1 + (cfg.img_size // cfg.patch_size) ** 2
        self.patch_embeddings = nn.Conv2d(
            in_channels=3, out_channels=cfg.embed_dim,
            kernel_size=cfg.patch_size, stride=cfg.patch_size
        )
        self.cls_token = nn.Parameter(torch.randn(1, 1, cfg.embed_dim))
        print(self.cls_token.shape)
        self.pos_embeddings = nn.Parameter(torch.randn(1, seq_len, cfg.embed_dim))
        self.dropout = nn.Dropout(cfg.p_drop)

    def forward(self, x):
        B = x.size()[0]
        x = self.patch_embeddings(x)  # [B, C, H, W]
        x = x.flatten(2).transpose(1, 2)  # [B, T, C]
        x = torch.cat([self.cls_token.expand(B, -1, -1), x], dim=1)
        print("x shape", x.shape)
        print("pos shape", self.pos_embeddings.shape)
        x = x + self.pos_embeddings
        x = self.dropout(x)
        return x
