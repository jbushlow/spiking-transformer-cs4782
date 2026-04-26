import torch
from torch.utils.data import Dataset
from datasets import load_dataset


class SST2Dataset(Dataset):
    def __init__(self, split='train'):
        data = load_dataset("glue", "sst2", split=split)
        self.texts = [x['sentence'] for x in data]
        self.labels = [x['label'] for x in data]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = list(self.texts[idx].encode('utf-8'))
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


class SST5Dataset(Dataset):
    def __init__(self, split='train'):
        data = load_dataset("SetFit/sst5", split=split)
        self.texts = [x['text'] for x in data]
        self.labels = [x['label'] for x in data]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = list(self.texts[idx].encode('utf-8'))
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


def collate_fn(batch, ctx_len=1024):
    tokens, labels = zip(*batch)
    max_len = min(max(t.size(0) for t in tokens), ctx_len)
    padded = torch.zeros(len(tokens), max_len, dtype=torch.long)
    for i, t in enumerate(tokens):
        t = t[:max_len]
        padded[i, :t.size(0)] = t
    return padded, torch.stack(labels)
