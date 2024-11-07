import os
import sys
import yaml
import argparse
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from loguru import logger

from src.model import GPT2
from src.utils import count_parameters
from src.configs import GPT2Config
from src.dataset import TextDataset


def parser(argv):
    parser = argparse.ArgumentParser(description='Training GPT2 model')
    parser.add_argument('--data_dir', type=str, default='data', help='path to data')
    parser.add_argument(
        '--output_dir', type=str, default='output', help='path to output')
    parser.add_argument(
        "--config_path", type=str, default="config.yaml",
        help="path to model configuration yaml file"
    )
    args = parser.parse_args(argv[1:])
    return args

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_float32_matmul_precision('high')  # 16k -> 19.5k

def main(args):
    config = yaml.load(open(args.config_path, 'r'), Loader=yaml.FullLoader)
    config = GPT2Config(**config)
    logger.info(config)
    
    model = GPT2(config.model)
    n_params = count_parameters(model)
    logger.info(f"Number of parameters: {n_params}")
    model.to(device)
    model = torch.compile(model)  # 19.5k -> 25k

    data_loader = TextDataset(
        data_dir=args.data_dir,
        batch_size=config.training.batch_size,
        seq_length=config.model.max_seq_len
    )

    optimizer = optim.AdamW(
        model.parameters(), lr=config.training.learning_rate,
        betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1,
        fused=True
    )

    batch_x, batch_y = data_loader.next_batch()
    batch_x = batch_x.to(device)
    batch_y = batch_y.to(device)
    logger.info(f"{batch_x.size()} {batch_y.size()}")
    logits = model(batch_x)
    logger.info(f"{logits.size()}")
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)), batch_y.view(-1)
    )
    logger.info(f"Loss: {loss.item()}")

    micro_batch_size = config.training.batch_size
    max_seq_len = config.model.max_seq_len
    total_batch_tokens = config.training.total_batch_tokens
    max_steps = int(1e10 // total_batch_tokens)
    grad_accum_steps = total_batch_tokens // (micro_batch_size * max_seq_len)
    logger.info(f"Max steps: {max_steps} Grad accum steps: {grad_accum_steps}")

    for step in range(max_steps):
        t_start = time.time()

        optimizer.zero_grad()

        for micro_step in range(grad_accum_steps):
            batch_x, batch_y = data_loader.next_batch()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            logits = model(batch_x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), batch_y.view(-1)
            )
            loss = loss / grad_accum_steps
            loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        t_end = time.time()
        #tokens_per_sec = batch_size * max_seq_len * grad_accum_steps / (t_end - t_start)
        tokens_per_sec = batch_x.size(0) * batch_x.size(1) * grad_accum_steps / (t_end - t_start)
        print(f"Step {step} Loss: {loss.item():.4f} Tokens/s: {tokens_per_sec}")


if __name__ == '__main__':
    args = parser(sys.argv)
    main(args)