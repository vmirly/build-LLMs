import os
import sys
import yaml
import argparse
import time

import torch
import torch.nn.functional as F
import torch.optim as optim

# ddp:
from torch.distributed import init_process_group, destroy_process_group
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

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

# for ddp, use torchrun:
# torchrun --nproc_per_node=4 train.py
def setup_ddp():
    if "LOCAL_RANK" in os.environ:
        use_ddp = True
        # initialize the process group
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        use_ddp = False
        rank = 0
        world_size = 1
    master_process = rank == 0
    return use_ddp, rank, world_size, master_process


def cleanup():
    dist.destroy_process_group()


def main(args):
    use_ddp, rank, world_size, master_process = setup_ddp()

    config = yaml.load(open(args.config_path, 'r'), Loader=yaml.FullLoader)
    config = GPT2Config(**config)
    if master_process:
        logger.info(config)
    
    model = GPT2(config.model)
    if master_process:
        n_params = count_parameters(model)
        logger.info(f"Number of parameters: {n_params}")
    model.to(device)
    model = torch.compile(model)  # 19.5k -> 25k
    if use_ddp:
        model = DistributedDataParallel(
            model, device_ids=[rank], output_device=rank
        )
        raw_model = model.module
    else:
        raw_model = model

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

    #batch_x, batch_y = data_loader.next_batch()
    #batch_x = batch_x.to(device)
    #batch_y = batch_y.to(device)
    #logger.info(f"{batch_x.size()} {batch_y.size()}")
    #logits = model(batch_x)
    #logger.info(f"{logits.size()}")
    #loss = F.cross_entropy(
    #    logits.view(-1, logits.size(-1)), batch_y.view(-1)
    #)
    #logger.info(f"Loss: {loss.item()}")

    B = config.training.batch_size
    T = config.model.max_seq_len
    total_batch_tokens = config.training.total_batch_tokens
    max_steps = int(1e10 // total_batch_tokens)
    grad_accum_steps = total_batch_tokens // (B * T)
    logger.info(
        f"Max steps: {max_steps} Grad accum steps: {grad_accum_steps}"
    )

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "log.txt"), "w") as f:
        pass

    for step in range(max_steps):
        t_start = time.time()

        optimizer.zero_grad()
        loss_train = 0.0
        for micro_step in range(grad_accum_steps):
            batch_x, batch_y = data_loader.next_batch()
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            if use_ddp:
                model.require_backward_grad_sync = (micro_step + 1 == grad_accum_steps)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(batch_x)
                loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)), batch_y.view(-1)
                )
            loss = loss / grad_accum_steps
            loss_train += loss.item()
            loss.backward()
        if use_ddp:
            # take average of loss across all GPUs
            dist.all_reduce(loss_train, op=dist.ReduceOp.AVG)
        optimizer.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_end = time.time()
        #tokens_per_sec = batch_size * max_seq_len * grad_accum_steps / (t_end - t_start)
        tokens_per_sec = batch_x.size(0) * batch_x.size(1) * grad_accum_steps / (t_end - t_start)
        print(f"Step {step} Loss: {loss_train:.4f} Tokens/s: {tokens_per_sec:.0f}")

        if (step + 1) % 1000 == 0 or (step + 1) == max_steps:
            # save checkpoint
            chekpoint_dict = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "step": step,
            }
            checkpoint_path = os.path.join(args.output_dir, f"checkpoint-{step}.pt")
            torch.save(chekpoint_dict, checkpoint_path)
            logger.info(f"Saved checkpoint at {checkpoint_path}")

        with open(os.path.join(args.output_dir, "log.txt"), "a") as f:
            f.write(f"{step + 1} {loss_train:.4f}\n")


if __name__ == '__main__':
    args = parser(sys.argv)
    main(args)