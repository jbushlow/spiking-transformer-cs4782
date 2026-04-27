import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from functools import partial

from config import get_spikegpt_46m_config
from nlu.nlu_model import SpikingGPTClassifier
from nlu.nlu_dataset import MRDataset, SST2Dataset, SST5Dataset, SubjDataset, collate_fn
from nlu.train_cls import evaluate

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


TASK_SPECS = {
    'sst2': {
        'dataset_cls': SST2Dataset,
        'split': 'validation',
        'num_classes': 2,
        'label': 'SST-2',
    },
    'sst5': {
        'dataset_cls': SST5Dataset,
        'split': 'test',
        'num_classes': 5,
        'label': 'SST-5',
    },
    'mr': {
        'dataset_cls': MRDataset,
        'split': 'test',
        'num_classes': 2,
        'label': 'MR',
    },
    'subj': {
        'dataset_cls': SubjDataset,
        'split': 'test',
        'num_classes': 2,
        'label': 'Subj',
    },
}


def test_task(task, checkpoint_path, batch_size=32, ctx_len=1024):
    spec = TASK_SPECS[task]
    config = get_spikegpt_46m_config(ctx_len=ctx_len)
    model = SpikingGPTClassifier(config, num_classes=spec['num_classes']).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint_path, map_location=DEVICE))

    loader = DataLoader(
        spec['dataset_cls'](split=spec['split']),
        batch_size=batch_size,
        collate_fn=partial(collate_fn, ctx_len=config.ctx_len),
    )
    acc = evaluate(model, loader, DEVICE, desc=task)
    print(f"{spec['label']} accuracy: {acc * 100:.2f}")
    return acc


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=sorted(TASK_SPECS.keys()), required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--ctx_len', type=int, default=1024)
    args = parser.parse_args()

    test_task(args.task, args.checkpoint, args.batch_size, args.ctx_len)
