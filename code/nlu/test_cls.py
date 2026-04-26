import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from functools import partial

from config import get_spikegpt_46m_config
from nlu.nlu_model import SpikingGPTClassifier
from nlu.nlu_dataset import SST2Dataset, SST5Dataset, collate_fn
from nlu.train_cls import evaluate

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def test_sst2(checkpoint_path, batch_size=32):
    config = get_spikegpt_46m_config()
    model = SpikingGPTClassifier(config, num_classes=2).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))

    loader = DataLoader(
        SST2Dataset(split='validation'), batch_size=batch_size,
        collate_fn=partial(collate_fn, ctx_len=config.ctx_len)
    )
    acc = evaluate(model, loader, DEVICE)
    print(f"SST-2 val accuracy: {acc:.4f}")
    return acc


def test_sst5(checkpoint_path, batch_size=32):
    config = get_spikegpt_46m_config()
    model = SpikingGPTClassifier(config, num_classes=5).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))

    loader = DataLoader(
        SST5Dataset(split='test'), batch_size=batch_size,
        collate_fn=partial(collate_fn, ctx_len=config.ctx_len)
    )
    acc = evaluate(model, loader, DEVICE)
    print(f"SST-5 test accuracy: {acc:.4f}")
    return acc


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=['sst2', 'sst5'], required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()

    if args.task == 'sst2':
        test_sst2(args.checkpoint, args.batch_size)
    else:
        test_sst5(args.checkpoint, args.batch_size)
