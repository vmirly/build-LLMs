import math

import torch
import torch.nn as nn
from torch.nn import functional as F

class CausalSelfAttention(nn.Module):
    """
    Causal self-attention layer, masking the future tokens.
    """
    def __init__(self, cfg):
        super().__init__()
        self.num_heads = cfg.num_heads
        self.q_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        self.k_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        self.v_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim)
        self.out_proj = nn.Linear(cfg.embed_dim, cfg.embed_dim)

        self.attn_dropout = nn.Dropout(0.1)
        self.resid_dropout = nn.Dropout(0.1)
        
        # Create a bias tensor to prevent attention to future tokens
        mask = torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len))
        self.register_buffer(
            'mask', (mask == 0).view(1, 1, cfg.max_seq_len, cfg.max_seq_len)
        )
        # mask will be a tensor like the following:
        # tensor([[[[False, True,  True,  ...,  True],
        #           [False, False, True,  ...,  True],
        #           [False, False, False, ...,  True],
        #           ...,
        #           [False, False, False, ..., False]]]])
        # where True values indicate that the token should be masked
        # i.e., replaced with -inf in the attention scores
        
    def forward(self, x):
        # Apply linear transformations to get queries, keys, and values
        # x: [B, T, C]
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        # q,k,v: [B, T, C]

        # Split the queries, keys, and values into multiple heads
        B, T, C = q.size()
        q = q.view(B, T, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        k = k.view(B, T, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        v = v.view(B, T, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        # q,k,v: [B, nh, T, C//nh]
        
        # Calculate attention scores
        scores = torch.matmul(q, k.permute(0, 1, 3, 2))
        scores = scores / (math.sqrt(k.size(-1)))
        scores.masked_fill_(self.mask[:, :, :T, :T], -torch.inf)
        # scores: [B, nh, T, T]
        
        # Apply softmax to get attention weights
        attn_weights = F.softmax(scores, dim=-1)
        # attn_weights: [B, nh, T, T]

        attn_weights = self.attn_dropout(attn_weights)
        
        # Multiply attention weights with values
        out = torch.matmul(attn_weights, v)
        # out: [B, nh, T, C//nh]

        # Concatenate the heads and apply a linear transformation
        out = out.permute(0, 2, 1, 3).contiguous().view(B, T, C)
        out = self.out_proj(out)
        # out: [B, T, C]

        out = self.resid_dropout(out)
        
        return out


class FeedForwardNetwork(nn.Module):
    """
    Feedforward network with GELU activation.
    """
    def __init__(self, cfg):
        super().__init__()

        embed_dim = cfg.embed_dim
        hidden_dim = cfg.embed_dim * 4
        p_drop = cfg.resid_dropout
        # Two linear layers with activation in between
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.gelu = nn.GELU(approximate='tanh')
        self.resid_dropout = nn.Dropout(p_drop)

    def forward(self, x):            # [B, T, C]
        x = self.gelu(self.fc1(x))  # [B, T, 2C]
        x = self.fc2(x)              # [B, T, C]
        x = self.resid_dropout(x)

        return x
    
class TransformerBlock(nn.Module):
    """
    Transformer block with a single self-attention layer
    and a feedforward network.
    """
    def __init__(self, config):
        super().__init__()
        self.mha = CausalSelfAttention(config)
        self.ln1 = nn.LayerNorm(config.embed_dim)
        self.ffn = FeedForwardNetwork(config)
        self.ln2 = nn.LayerNorm(config.embed_dim)

        self.resid_dropout = nn.Dropout(config.resid_dropout)

    def forward(self, x):
        # Apply self-attention and add residual connection
        shortcut = x
        x = self.ln1(x)
        x = self.mha(x)[0]
        x = self.resid_dropout(x)
        x = shortcut + x

        # Apply feedforward network and add residual connection
        shortcut = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = self.resid_dropout(x)
        x = shortcut + x

        return x


class GPT2(nn.Module):
    """
    GPT-2 model with a transformer decoder.
    """
    def __init__(self, config):
        super().__init__()
        embed_dim = config.embed_dim
        vocab_size = config.vocab_size
        context_length = config.max_seq_len
        self.num_layers = config.num_layers

        self.token_emb = nn.Embedding(vocab_size, embed_dim)
        self.pos_emb = nn.Embedding(context_length, embed_dim)
        self.embed_dropout = nn.Dropout(config.embed_dropout)

        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(self.num_layers)
        ])

        self.ln_final = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.head.weight = self.token_emb.weight

        # initialize the weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            std = 0.02 / math.sqrt(2 * self.num_layers)
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx):
        # Generate token embeddings
        tok_emb = self.token_emb(idx)
        # Generate position embeddings
        pos = torch.arange(idx.size(1), device=idx.device).unsqueeze(0)
        pos_emb = self.pos_emb(pos)
        x = self.embed_dropout(tok_emb + pos_emb)

        # Apply the transformer blocks
        for layer in self.layers:
            x = layer(x)

        # Apply the final layer norm
        x = self.ln_final(x)

        # Generate logits
        logits = self.head(x)

        return logits
