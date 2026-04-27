import sys
import os
from pathlib import Path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from functools import partial
from tqdm import tqdm
from spikingjelly.activation_based import functional


def train_epoch(model, loader, optimizer, device, epoch, max_epochs):
    model.train()
    total_loss = 0.0
    bar = tqdm(loader, desc=f"epoch {epoch}/{max_epochs} [train]", leave=False)
    for tokens, labels in bar:
        tokens, labels = tokens.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = model(tokens, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        functional.reset_net(model)
        total_loss += loss.item()
        bar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / len(loader)


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


def train(model, train_dataset, val_dataset, trainer_config, device, ctx_len=1024):
    from nlu.nlu_dataset import collate_fn

    collate = partial(collate_fn, ctx_len=ctx_len)
    train_loader = DataLoader(
        train_dataset, batch_size=trainer_config.batch_size,
        shuffle=True, collate_fn=collate
    )
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

    for epoch in range(1, trainer_config.max_epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, device, epoch, trainer_config.max_epochs)
        acc = evaluate(model, val_loader, device)
        print(f"epoch {epoch}/{trainer_config.max_epochs}: loss={loss:.4f}  val_acc={acc:.4f}")

        latest_path = checkpoint_dir / "latest.pt"
        torch.save(model.state_dict(), latest_path)

        epoch_path = checkpoint_dir / f"epoch_{epoch}.pt"
        torch.save(model.state_dict(), epoch_path)

        if acc > best_acc:
            best_acc = acc
            best_path = checkpoint_dir / "best.pt"
            torch.save(model.state_dict(), best_path)
