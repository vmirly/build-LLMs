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

def main(args):
    config = yaml.load(open(args.config_path, 'r'), Loader=yaml.FullLoader)
    config = GPT2Config(**config)
    logger.info(config)

    model = GPT2(config.model)
    n_params = count_parameters(model)
    logger.info(f"Number of parameters: {n_params}")
    model.to(device)

    data_loader = TextDataset(
        data_dir=args.data_dir,
        batch_size=config.training.batch_size,
        seq_length=config.model.max_seq_len
    )

    optimizer = optim.AdamW(
        model.parameters(), lr=config.training.learning_rate,
        betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1
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
    for epoch in range(100):
        t_start = time.time()
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)), batch_y.view(-1)
        )
        loss.backward()
        optimizer.step()
        torch.cuda.synchronize()
        t_end = time.time()
        tokens_per_sec = (batch_x.size(0) * batch_x.size(1)) / (t_end - t_start)
        logger.info(f"Epoch {epoch} Loss: {loss.item()} Tokens/s: {tokens_per_sec}")




if __name__ == '__main__':
    args = parser(sys.argv)
    main(args)