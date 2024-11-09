# Description: Utility functions for GPT2 model
import math
import torch


def load_hf_weights(model, hf_model):
    """
    Load the weights from a Hugging Face model into our model.
    """
    # Load the weights from the Hugging Face model
    hf_dict = hf_model.state_dict()

    # assign emnbedding weights
    model.token_emb.weight.data.copy_(hf_dict['wte.weight'])
    model.pos_emb.weight.data.copy_(hf_dict['wpe.weight'])

    # assign transformer block weights
    for idx, layer in enumerate(model.layers):
        # MHA weights and biases
        qkv_weight = hf_dict[f'h.{idx}.attn.c_attn.weight']
        qkv_bias = hf_dict[f'h.{idx}.attn.c_attn.bias']
        q_weight, k_weight, v_weight = qkv_weight.chunk(3, dim=1)
        q_bias, k_bias, v_bias = qkv_bias.chunk(3)
        layer.mha.q_proj.weight.data.copy_(q_weight.T)
        layer.mha.q_proj.bias.data.copy_(q_bias)
        layer.mha.k_proj.weight.data.copy_(k_weight.T)
        layer.mha.k_proj.bias.data.copy_(k_bias)
        layer.mha.v_proj.weight.data.copy_(v_weight.T)
        layer.mha.v_proj.bias.data.copy_(v_bias)
        # MHA out projection weights and biases
        c_weight = hf_dict[f'h.{idx}.attn.c_proj.weight']
        c_bias = hf_dict[f'h.{idx}.attn.c_proj.bias']
        layer.mha.out_proj.weight.data.copy_(c_weight.T)
        layer.mha.out_proj.bias.data.copy_(c_bias)
        # Layer norm weights and biases
        layer.ln1.weight.data.copy_(hf_dict[f'h.{idx}.ln_1.weight'])
        layer.ln1.bias.data.copy_(hf_dict[f'h.{idx}.ln_1.bias'])
        layer.ln2.weight.data.copy_(hf_dict[f'h.{idx}.ln_2.weight'])
        layer.ln2.bias.data.copy_(hf_dict[f'h.{idx}.ln_2.bias'])
        # FFN weights and biases
        layer.ffn.fc1.weight.data.copy_(hf_dict[f'h.{idx}.mlp.c_fc.weight'].T)
        layer.ffn.fc1.bias.data.copy_(hf_dict[f'h.{idx}.mlp.c_fc.bias'])
        layer.ffn.fc2.weight.data.copy_(hf_dict[f'h.{idx}.mlp.c_proj.weight'].T)
        layer.ffn.fc2.bias.data.copy_(hf_dict[f'h.{idx}.mlp.c_proj.bias'])

    # assign final layer norm weights
    model.ln_final.weight.data.copy_(hf_dict['ln_f.weight'])
    model.ln_final.bias.data.copy_(hf_dict['ln_f.bias'])
    # assign head weights (wte)
    #model.head.weight.data.copy_(hf_dict['wte.weight'])


def generate_text(
        model,
        tokenizer,
        prompt,
        max_len=100,
        temperature=1.0,
    ):
    model.eval()
    prompt = tokenizer.encode(prompt)
    prompt = torch.tensor(prompt).unsqueeze(0)
    generated = prompt
    with torch.no_grad():
        for _ in range(max_len):
            logits = model(generated)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            generated = torch.cat((generated, next_token), dim=1)

    return tokenizer.decode(generated[0].tolist())


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


class CosineLRScheduler():
    def __init__(
        self,
        optimizer,
        max_lr,
        min_lr,
        warmup_steps,
        total_steps
    ):
        self.optimizer = optimizer
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
    
    def get_lr(self, step):
        if step < self.warmup_steps:
            return self.max_lr * (step + 1) / self.warmup_steps
        else:
            x = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            alpha = 0.5 * (1.0 + math.cos(math.pi * x))
            return self.min_lr + alpha * (self.max_lr - self.min_lr)
        
    def update_lr(self, step):
        lr = self.get_lr(step)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
