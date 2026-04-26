from pathlib import Path
import re

import torch


_EPOCH_CHECKPOINT_RE = re.compile(r"epoch_(\d+)\.pt$")


def _checkpoint_sort_key(path):
    epoch_match = _EPOCH_CHECKPOINT_RE.match(path.name)
    if epoch_match:
        return int(epoch_match.group(1))

    return -1


def find_latest_checkpoint(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return None

    checkpoints = [path for path in checkpoint_dir.glob("*.pt") if _checkpoint_sort_key(path) >= 0]
    if not checkpoints:
        return None

    return max(checkpoints, key=_checkpoint_sort_key)


def save_checkpoint(path, state):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_checkpoint(path, device):
    return torch.load(Path(path), map_location=device)
