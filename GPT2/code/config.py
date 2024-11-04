# Configuration file for the GPT2 model
from pydantic import BaseModel


class ModelConfig(BaseModel):
    hf_model_name: str = ""
    vocab_size: int = 50257
    max_seq_len: int = 1024
    embed_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    attn_dropout: float = 0.1
    resid_dropout: float = 0.1
    embed_dropout: float = 0.1