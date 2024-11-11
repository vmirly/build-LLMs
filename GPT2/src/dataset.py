import os
import random

import torch
import numpy as np



class TextDataset():
    def __init__(
            self, 
            data_dir,
            batch_size,
            seq_length,
            process_rank=0,
            num_processes=1
        ):

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.data_dir = data_dir
        self.process_rank = process_rank
        self.num_processes = num_processes

        self.shard_files = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
        ]
        random.shuffle(self.shard_files)

        self.current_shard = 0
        self.current_index = 0
        self.shard = self.load_shard(self.current_shard)

    def load_shard(self, shard_idx):
        shard_file = self.shard_files[shard_idx]
        data = np.load(shard_file).astype(np.int32)
        data = torch.tensor(data, dtype=torch.long)
        return data

    def next_batch(self):
        B, T = self.batch_size, self.seq_length
        np = self.num_processes
        if self.current_index + B * T * np + 1 >= len(self.shard):
            if self.current_shard == len(self.shard_files) - 1:
                # reshuffle shard files
                random.shuffle(self.shard_files)
            self.current_shard = (self.current_shard + 1) % len(self.shard_files)
            self.shard = self.load_shard(self.current_shard)
            self.current_index = B * T * self.process_rank

        idx = self.current_index
        total = B * T * np
        x = self.shard[idx: idx + total]
        y = self.shard[idx + 1: idx + total + 1]
        self.current_index += total

        x = x.view(self.batch_size, self.seq_length)
        y = y.view(self.batch_size, self.seq_length)

        return x, y


# testing
if __name__ == "__main__":
    data_dir = "data"
    batch_size = 8
    seq_length = 32
    dataset = TextDataset(data_dir, batch_size, seq_length)
    x, y = dataset.next_batch()
    print(x.shape, y.shape)
    print(x[0])
    print(y[0])
