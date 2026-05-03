import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import math
import torch
from datetime import datetime
from pathlib import Path
from torch.utils.data import DataLoader
from functools import partial

from config import get_spikegpt_46m_config
from nlu.nlu_model import SpikingGPTClassifier
from nlu.nlu_dataset import MRDataset, SST2Dataset, SST5Dataset, SubjDataset, collate_fn
from nlu.train_cls import evaluate

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
RESULTS_PATH = Path(__file__).resolve().parent.parent.parent / 'results' / 'test_results.json'


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


def load_results():
    if RESULTS_PATH.exists():
        return json.loads(RESULTS_PATH.read_text())
    return {}


def save_results(results):
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_PATH.write_text(json.dumps(results, indent=2))


def test_task(task, checkpoint_path, batch_size=32, ctx_len=1024):
    spec = TASK_SPECS[task]
    config = get_spikegpt_46m_config(ctx_len=ctx_len)
    model = SpikingGPTClassifier(config, num_classes=spec['num_classes']).to(DEVICE)
    ckpt = torch.load(checkpoint_path, map_location=DEVICE)
    model.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)

    loader = DataLoader(
        spec['dataset_cls'](split=spec['split']),
        batch_size=batch_size,
        collate_fn=partial(collate_fn, ctx_len=config.ctx_len),
    )
    acc, loss = evaluate(model, loader, DEVICE, desc=task)
    print(f"{spec['label']} accuracy: {acc * 100:.2f}%  loss: {loss:.4f}")

    results = load_results()
    results[task] = {
        'label': spec['label'],
        'accuracy': round(acc * 100, 4),
        'loss': round(loss, 6),
        'checkpoint': str(checkpoint_path),
        'ctx_len': ctx_len,
        'timestamp': datetime.now().isoformat(),
    }

    # Pull in LM results from test.py's summary if available
    lm_summary = Path(__file__).resolve().parent.parent.parent / 'results' / 'latex_tables' / 'metrics_summary.txt'
    if lm_summary.exists() and 'lm' not in results:
        lm = {}
        for line in lm_summary.read_text().splitlines():
            if '=' in line:
                k, v = line.split('=', 1)
                lm[k.strip()] = v.strip()
        results['lm'] = {
            'test_bpc': lm.get('test_bpc', 'not_computed'),
            'test_loss': lm.get('test_loss', 'not_computed'),
            'train_bpc': lm.get('train_bpc', 'not_computed'),
            'checkpoint': lm.get('checkpoint', 'unknown'),
        }

    save_results(results)
    print(f"Results saved to {RESULTS_PATH}")
    print_summary(results)
    return acc, loss


def print_summary(results):
    print("\n=== Test Results Summary ===")
    if 'lm' in results:
        lm = results['lm']
        print(f"  Language Model  — test_bpc: {lm['test_bpc']}  test_loss: {lm['test_loss']}")
    for key, spec in TASK_SPECS.items():
        if key in results:
            r = results[key]
            print(f"  {spec['label']:8s}        — accuracy: {r['accuracy']:.2f}%  loss: {r['loss']:.4f}")
    print()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', choices=sorted(TASK_SPECS.keys()), required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--ctx_len', type=int, default=1024)
    args = parser.parse_args()

    test_task(args.task, args.checkpoint, args.batch_size, args.ctx_len)
