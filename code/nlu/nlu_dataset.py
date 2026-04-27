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


class MRDataset(Dataset):
    def __init__(self, split='train'):
        if split == 'validation':
            split = 'test'
        data = load_dataset("sh0416/mr", split=split)
        text_key = 'text' if 'text' in data.column_names else data.column_names[0]
        label_key = 'label' if 'label' in data.column_names else data.column_names[1]
        self.texts = [x[text_key] for x in data]
        self.labels = [x[label_key] for x in data]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = list(self.texts[idx].encode('utf-8'))
        return torch.tensor(encoded, dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)


class SubjDataset(Dataset):
    _SPLITS = None

    @classmethod
    def _get_splits(cls):
        if cls._SPLITS is None:
            data = load_dataset("SetFit/subj", split="train")
            train_and_holdout = data.train_test_split(test_size=0.2, seed=42)
            val_and_test = train_and_holdout["test"].train_test_split(test_size=0.5, seed=42)
            cls._SPLITS = {
                "train": train_and_holdout["train"],
                "validation": val_and_test["train"],
                "test": val_and_test["test"],
            }
        return cls._SPLITS

    def __init__(self, split='train'):
        data = self._get_splits()[split]
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
