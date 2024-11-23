# config file for ViT
from pydantic import BaseModel

class ModelConfig(BaseModel):
    embed_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    p_drop: float = 0.1
    patch_size: int = 16
    num_classes: int = 1000
    img_size: int = 224
    num_channels: int = 3
