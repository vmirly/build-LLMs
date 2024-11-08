Build a GPT2 from Scratch
====

### Series of optimization:

 * Tokens/s is based on NVIDIA RTX 3090Ti

| Optimization               | Tokens/s    |
|----------------------------|-------------|
| Baseline                   | 17048.7     |
| TF32                       | 20597.4     |
| Compile                    | 27216.3     |
| Combined-qkv-proj          | 28582.9     |
| Flash Attention            | 28044.8     |
| Without Flash Attn         | 28834.2     |
| bfloat16 (same batch size) | 49004       |
| Flash Attn + bfloat16      | 61324.4     |
| Increase Batch size (16)   | 67670.9     |
| Gradient Accumulation      | 69163.8     |


