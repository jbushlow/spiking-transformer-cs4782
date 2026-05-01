import torch
from torch.utils.data import Dataset
from pathlib import Path


class AsciiArtDataset(Dataset):
    """Byte-level ASCII art dataset — same interface as Enwik8Dataset."""

    def __init__(self, file_path, ctx_len=512, split='train', val_ratio=0.05):
        data_bytes = Path(file_path).read_bytes()
        data = torch.tensor(list(data_bytes), dtype=torch.long)

        n = len(data)
        n_val = max(1, int(n * val_ratio))
        if split == 'train':
            self.data = data[n_val:]
        elif split == 'val':
            self.data = data[:n_val]
        else:
            self.data = data

        self.ctx_len = ctx_len

    def __len__(self):
        return max(0, len(self.data) - self.ctx_len)

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.ctx_len]
        y = self.data[idx + 1 : idx + self.ctx_len + 1]
        return x, y
