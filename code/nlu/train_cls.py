import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader, Subset
from functools import partial
from tqdm import tqdm
from spikingjelly.activation_based import functional


def _make_epoch_loader(dataset, batch_size, collate_fn, epoch, seed=42, skip_steps=0):
    # Reproducible per-epoch shuffle so we can resume mid-epoch with the same order
    g = torch.Generator()
    g.manual_seed(seed + epoch)
    indices = torch.randperm(len(dataset), generator=g).tolist()
    if skip_steps > 0:
        indices = indices[skip_steps * batch_size:]
    return DataLoader(Subset(dataset, indices), batch_size=batch_size, shuffle=False, collate_fn=collate_fn)


def train_epoch(model, dataset, batch_size, collate_fn, optimizer, device,
                epoch, max_epochs, save_steps=0, checkpoint_dir=None, start_step=0, seed=42):
    loader = _make_epoch_loader(dataset, batch_size, collate_fn, epoch, seed=seed, skip_steps=start_step)
    model.train()
    total_loss = 0.0
    n_batches = 0

    bar = tqdm(loader, desc=f"epoch {epoch}/{max_epochs} [train]", leave=False)
    for step_offset, (tokens, labels) in enumerate(bar):
        global_step = start_step + step_offset
        tokens, labels = tokens.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = model(tokens, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        functional.reset_net(model)
        total_loss += loss.item()
        n_batches += 1
        bar.set_postfix(loss=f"{loss.item():.4f}", step=global_step + 1)

        if save_steps > 0 and checkpoint_dir is not None and (global_step + 1) % save_steps == 0:
            torch.save({
                'epoch': epoch,
                'step': global_step + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, Path(checkpoint_dir) / 'resume.pt')

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader, device, desc="val"):
    model.eval()
    correct = total = 0
    for tokens, labels in tqdm(loader, desc=f"  [{desc}]", leave=False):
        tokens, labels = tokens.to(device), labels.to(device)
        logits = model(tokens)
        functional.reset_net(model)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


def train(model, train_dataset, val_dataset, trainer_config, device, ctx_len=1024, save_steps=500, seed=42):
    from nlu.nlu_dataset import collate_fn as _collate_fn

    collate = partial(_collate_fn, ctx_len=ctx_len)
    val_loader = DataLoader(
        val_dataset, batch_size=trainer_config.batch_size,
        shuffle=False, collate_fn=collate
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=trainer_config.learning_rate,
        betas=trainer_config.betas,
        eps=trainer_config.eps,
    )

    checkpoint_dir = Path(getattr(trainer_config, "cls_checkpoint_dir", "results/nlu_checkpoints"))
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_acc = -1.0
    start_epoch = 1
    start_step = 0

    resume_path = checkpoint_dir / 'resume.pt'
    if resume_path.exists():
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch']
        start_step = ckpt['step']
        print(f"Resuming from epoch {start_epoch}, step {start_step}")

    for epoch in range(start_epoch, trainer_config.max_epochs + 1):
        step_offset = start_step if epoch == start_epoch else 0
        loss = train_epoch(
            model, train_dataset, trainer_config.batch_size, collate, optimizer, device,
            epoch, trainer_config.max_epochs,
            save_steps=save_steps, checkpoint_dir=checkpoint_dir,
            start_step=step_offset, seed=seed,
        )
        start_step = 0  # only skip for the resumed epoch

        acc = evaluate(model, val_loader, device)
        print(f"epoch {epoch}/{trainer_config.max_epochs}: loss={loss:.4f}  val_acc={acc:.4f}")

        epoch_ckpt = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }
        torch.save(epoch_ckpt, checkpoint_dir / 'latest.pt')
        torch.save(epoch_ckpt, checkpoint_dir / f'epoch_{epoch}.pt')

        if acc > best_acc:
            best_acc = acc
            torch.save(epoch_ckpt, checkpoint_dir / 'best.pt')

        # Remove mid-epoch resume checkpoint once the full epoch is saved
        if resume_path.exists():
            resume_path.unlink()
