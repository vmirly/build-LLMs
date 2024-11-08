import os
import torch
import numpy as np



class TextDataset():
    def __init__(
            self, 
            data_dir,
            batch_size,
            seq_length,
        ):

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.data_dir = data_dir

        self.shard_files = [
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
        ]

        self.current_shard = 0
        self.current_index = 0
        self.shard = self.load_shard(self.current_shard)

    def load_shard(self, shard_idx):
        shard_file = self.shard_files[shard_idx]
        data = np.load(shard_file).astype(np.int32)
        data = torch.tensor(data, dtype=torch.long)
        return data

    def next_batch(self):
        if self.current_index + self.seq_length + 1 >= len(self.shard):
            self.current_shard = (self.current_shard + 1) % len(self.shard_files)
            self.shard = self.load_shard(self.current_shard)
            self.current_index = 0

        idx = self.current_index
        total = self.batch_size * self.seq_length
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
