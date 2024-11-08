# Configuration file for the GPT2 model
from pydantic import BaseModel


class ModelConfig(BaseModel):
    hf_model_name: str = ""
    vocab_size: int = 50257
    max_seq_len: int = 1024
    embed_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    p_drop_attn: float = 0.1
    p_drop_resid: float = 0.1
    p_drop_embed: float = 0.1


class Training(BaseModel):
    batch_size: int = 4
    total_batch_tokens: int = 4 * 1024
    seq_length: int = 1024
    epochs: int = 10
    learning_rate: float = 1e-4
    warmup_steps: int = 1000
    gradient_accumulation_steps: int = 1
    max_norm: float = 1.0


class GPT2Config(BaseModel):
    model: ModelConfig
    training: Training
