from pathlib import Path
import re

import torch


_EPOCH_CHECKPOINT_RE = re.compile(r"epoch_(\d+)\.pt$")
_STEP_CHECKPOINT_RE = re.compile(r"epoch_(\d+)_step_(\d+)\.pt$")


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


def prune_checkpoints(checkpoint_dir, keep_last_epoch_checkpoints=5, keep_last_step_checkpoints=3):
    checkpoint_dir = Path(checkpoint_dir)
    if not checkpoint_dir.exists():
        return

    epoch_paths = []
    step_paths = []

    for path in checkpoint_dir.glob("*.pt"):
        epoch_match = _EPOCH_CHECKPOINT_RE.match(path.name)
        if epoch_match:
            epoch_paths.append((int(epoch_match.group(1)), path))
            continue

        step_match = _STEP_CHECKPOINT_RE.match(path.name)
        if step_match:
            epoch_num, step_num = step_match.groups()
            step_paths.append((int(epoch_num), int(step_num), path))

    epoch_paths.sort()
    step_paths.sort()

    epoch_paths_to_keep = {
        path for _, path in epoch_paths[-keep_last_epoch_checkpoints:]
    } if keep_last_epoch_checkpoints > 0 else set()

    step_paths_to_keep = {
        path for _, _, path in step_paths[-keep_last_step_checkpoints:]
    } if keep_last_step_checkpoints > 0 else set()

    for _, path in epoch_paths:
        if path not in epoch_paths_to_keep and path.exists():
            path.unlink()

    for _, _, path in step_paths:
        if path not in step_paths_to_keep and path.exists():
            path.unlink()
