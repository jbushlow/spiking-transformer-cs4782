import torch 
from torch.utils.data import Dataset

class Enwik8Dataset(Dataset):
    def __init__(self, path, ctx_len):
        with open(path, 'rb') as f:
            data = f.read()
        self.data = torch.tensor(list(data), dtype=torch.long)
        self.ctx_len = ctx_len

    def __len__(self):
        return len(self.data) - self.ctx_len
    
    def __getitem__(self, idx):
        x = self.data[idx:idx+self.ctx_len]
        y = self.data[idx+1:idx+self.ctx_len+1]
        return x, y
