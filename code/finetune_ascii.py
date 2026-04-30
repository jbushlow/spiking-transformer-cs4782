"""
Fine-tune SpikingGPT on ASCII art data, starting from a pretrained checkpoint.

CPU-recommended command (≈2 hrs):
  python code/finetune_ascii.py --pretrained epoch_185.pt \\
      --batch-size 2 --ctx-len 256 --steps-per-epoch 50 --epochs 20

GPU command (≈15 min):
  python code/finetune_ascii.py --pretrained epoch_185.pt

Resume after interruption:
  python code/finetune_ascii.py --pretrained epoch_185.pt --resume
"""
import argparse
import math
import random
import re
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from spikingjelly.activation_based import functional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ascii_dataset import AsciiArtDataset
from config import GPTConfig
from model import SpikingGPT
from utils.checkpoint import (
    load_checkpoint,
    prune_checkpoints,
    save_checkpoint,
)

MAX_VAL_STEPS = 20


def parse_args():
    p = argparse.ArgumentParser(description='Fine-tune SpikingGPT on ASCII art')
    p.add_argument('--pretrained', required=True,
                   help='Pretrained checkpoint (e.g. epoch_185.pt)')
    p.add_argument('--data', default=None,
                   help='ASCII art training file (default: data/ascii_art/train.txt)')
    p.add_argument('--ctx-len',         type=int,   default=512)
    p.add_argument('--epochs',          type=int,   default=30)
    p.add_argument('--batch-size',      type=int,   default=8)
    p.add_argument('--steps-per-epoch', type=int,   default=200,
                   help='Max train steps per epoch. Use 50 on CPU to keep epochs short.')
    p.add_argument('--save-every',      type=int,   default=25,
                   help='Save a step checkpoint every N steps (crash recovery).')
    p.add_argument('--lr',              type=float, default=3e-5)
    p.add_argument('--lr-final',        type=float, default=1e-6)
    p.add_argument('--checkpoint-dir',  default='results/ascii_checkpoints')
    p.add_argument('--resume', action='store_true',
                   help='Resume from latest checkpoint in --checkpoint-dir')
    return p.parse_args()


def set_seed(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def cosine_lr(step, total_steps, lr_init, lr_final):
    progress = min(step / max(total_steps, 1), 1.0)
    return lr_final + (lr_init - lr_final) * 0.5 * (1.0 + math.cos(math.pi * progress))


def evaluate(model, loader, device):
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            loss = model(x, y)
            total_loss += loss.item()
            functional.reset_net(model)
            n += 1
            if n >= MAX_VAL_STEPS:
                break
    return total_loss / max(n, 1)


def find_latest_checkpoint(ckpt_dir):
    """Return the most recent checkpoint (prefers step > epoch checkpoints)."""
    ckpt_dir = Path(ckpt_dir)
    if not ckpt_dir.exists():
        return None, None, None

    best_path = None
    best_epoch, best_step = -1, -1

    epoch_re = re.compile(r'epoch_(\d+)\.pt$')
    step_re  = re.compile(r'epoch_(\d+)_step_(\d+)\.pt$')

    for p in ckpt_dir.glob('*.pt'):
        m = step_re.match(p.name)
        if m:
            ep, st = int(m.group(1)), int(m.group(2))
            if (ep, st) > (best_epoch, best_step):
                best_epoch, best_step, best_path = ep, st, p
            continue
        m = epoch_re.match(p.name)
        if m:
            ep = int(m.group(1))
            if (ep, -1) > (best_epoch, best_step):
                best_epoch, best_step, best_path = ep, -1, p

    return best_path, best_epoch, best_step


def main():
    args = parse_args()
    set_seed()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'Device: {device}')

    # ── Load pretrained model ──────────────────────────────────────────────
    print(f'Loading pretrained checkpoint: {args.pretrained}')
    ckpt = torch.load(args.pretrained, map_location=device, weights_only=False)
    saved_cfg = ckpt.get('config', ckpt.get('model_config', {}))

    model_config = GPTConfig(**{k: v for k, v in saved_cfg.items() if k != 'n_ffn'})
    model_config.ctx_len = args.ctx_len

    model = SpikingGPT(model_config).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    print(f'  pretrained epoch={ckpt.get("epoch","?")}  '
          f'val_loss={ckpt.get("val_loss", float("nan")):.4f}')

    # ── Dataset ────────────────────────────────────────────────────────────
    project_root = Path(__file__).resolve().parent.parent
    data_path = Path(args.data) if args.data else project_root / 'data' / 'ascii_art' / 'train.txt'
    if not data_path.exists():
        print(f'ERROR: data file not found: {data_path}')
        print('Run first:  python code/prepare_ascii_data.py')
        raise SystemExit(1)

    train_ds = AsciiArtDataset(data_path, ctx_len=args.ctx_len, split='train')
    val_ds   = AsciiArtDataset(data_path, ctx_len=args.ctx_len, split='val')
    print(f'Data: train={len(train_ds):,}  val={len(val_ds):,}  ctx_len={args.ctx_len}')

    if len(train_ds) == 0:
        print('ERROR: dataset is empty — is the data file populated?')
        raise SystemExit(1)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    # ── Optimizer ──────────────────────────────────────────────────────────
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), eps=4e-9)

    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ── Resume ─────────────────────────────────────────────────────────────
    start_epoch, resume_step = 0, 0
    if args.resume:
        latest_path, latest_epoch, latest_step = find_latest_checkpoint(ckpt_dir)
        if latest_path:
            resume_ckpt = load_checkpoint(latest_path, device)
            model.load_state_dict(resume_ckpt['model_state_dict'])
            optimizer.load_state_dict(resume_ckpt['optimizer_state_dict'])
            start_epoch  = resume_ckpt.get('epoch', latest_epoch)
            resume_step  = resume_ckpt.get('step_in_epoch', 0)
            if latest_step >= 0:
                # Step checkpoint: we're mid-epoch, resume at start_epoch - 1
                start_epoch -= 1
            print(f'Resumed from {latest_path.name} '
                  f'(epoch {start_epoch+1}, step {resume_step})')
        else:
            print('No checkpoint found — starting from pretrained.')

    total_steps = args.epochs * args.steps_per_epoch
    global_step = start_epoch * args.steps_per_epoch + resume_step

    # ── Training loop ──────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        model.train()
        total_loss, n_batches = 0.0, 0
        skip_steps = resume_step if epoch == start_epoch else 0

        for step, (x, y) in enumerate(train_loader):
            if step < skip_steps:
                continue

            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            loss = model(x, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            functional.reset_net(model)

            lr = cosine_lr(global_step, total_steps, args.lr, args.lr_final)
            for pg in optimizer.param_groups:
                pg['lr'] = lr

            total_loss += loss.item()
            n_batches  += 1
            global_step += 1

            if n_batches % 10 == 0:
                print(f'  epoch {epoch+1} step {n_batches:4d} | '
                      f'loss={loss.item():.4f}  lr={lr:.2e}')

            # Step-level checkpoint for crash recovery
            if args.save_every > 0 and n_batches % args.save_every == 0:
                step_path = ckpt_dir / f'epoch_{epoch+1}_step_{n_batches}.pt'
                save_checkpoint(step_path, {
                    'epoch':                epoch + 1,
                    'step_in_epoch':        n_batches,
                    'model_state_dict':     model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'train_loss':           total_loss / n_batches,
                    'config':               vars(model_config),
                })
                prune_checkpoints(ckpt_dir, keep_last_epoch_checkpoints=3,
                                  keep_last_step_checkpoints=2)

            if n_batches >= args.steps_per_epoch:
                break

        train_loss = total_loss / max(n_batches, 1)
        val_loss   = evaluate(model, val_loader, device)
        print(f'Epoch {epoch+1}/{args.epochs} — train={train_loss:.4f}  val={val_loss:.4f}')

        epoch_path = ckpt_dir / f'epoch_{epoch+1}.pt'
        save_checkpoint(epoch_path, {
            'epoch':                epoch + 1,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss':           train_loss,
            'val_loss':             val_loss,
            'config':               vars(model_config),
            'finetune_args':        vars(args),
        })
        prune_checkpoints(ckpt_dir, keep_last_epoch_checkpoints=3,
                          keep_last_step_checkpoints=2)
        print(f'  → saved {epoch_path.name}')


if __name__ == '__main__':
    main()
