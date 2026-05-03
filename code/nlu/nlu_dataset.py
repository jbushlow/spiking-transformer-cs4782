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
        hf_split = 'test' if split == 'validation' else 'train'
        # sh0416/mr CSVs have no header row; load each file directly with explicit column names
        data = load_dataset(
            "csv",
            data_files=f"hf://datasets/sh0416/mr/{hf_split}.csv",
            column_names=["text", "label"],
            header=None,
            split="train",
        )
        self.texts = [x["text"] for x in data]
        self.labels = [int(x["label"]) for x in data]

        unique = set(self.labels)
        counts = {v: self.labels.count(v) for v in sorted(unique)}
        print(f"[MRDataset/{hf_split}] {len(self.texts)} samples, label distribution: {counts}")
        print(f"  sample[0]: label={self.labels[0]}  text={self.texts[0][:80]!r}")
        print(f"  sample[1]: label={self.labels[1]}  text={self.texts[1][:80]!r}")

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
