import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from functools import partial
from spikingjelly.activation_based import functional


def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0.0
    for tokens, labels in loader:
        tokens, labels = tokens.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = model(tokens, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        functional.reset_net(model)
        total_loss += loss.item()
    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for tokens, labels in loader:
        tokens, labels = tokens.to(device), labels.to(device)
        logits = model(tokens)
        functional.reset_net(model)
        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct / total


def train(model, train_dataset, val_dataset, trainer_config, device, ctx_len=1024):
    from nlu_dataset import collate_fn

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

    for epoch in range(1, trainer_config.max_epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, device)
        acc = evaluate(model, val_loader, device)
        print(f"epoch {epoch}: loss={loss:.4f}  val_acc={acc:.4f}")
