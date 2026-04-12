from pathlib import Path
import torch
from torch.utils.data import DataLoader
from spikingjelly.activation_based import functional

from dataset import Enwik8Dataset
from config import GPTConfig, TrainerConfig
from model import SpikingGPT


def evaluate(model, loader, device):
    model.eval()
    total_loss = 0.0
    total_batches = 0

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            loss = model(x, y)
            total_loss += loss.item()
            total_batches += 1

            functional.reset_net(model)

    return total_loss / max(total_batches, 1)


def main():
    model_config = GPTConfig(
        ctx_len=256,
        n_embd=128,
        n_layer=2,
    )
    trainer_config = TrainerConfig(
        max_epochs=2,
        batch_size=4,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("using device:", device)

    data_dir = Path(__file__).resolve().parent.parent / "data" / "enwik8_split"

    train_dataset = Enwik8Dataset(data_dir / "train.txt", model_config.ctx_len)
    val_dataset = Enwik8Dataset(data_dir / "valid.txt", model_config.ctx_len)

    train_loader = DataLoader(
        train_dataset,
        batch_size=trainer_config.batch_size,
        shuffle=True,
        num_workers=trainer_config.num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=trainer_config.batch_size,
        shuffle=False,
        num_workers=trainer_config.num_workers,
    )

    model = SpikingGPT(model_config).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=trainer_config.learning_rate,
        betas=trainer_config.betas,
        eps=trainer_config.eps,
    )



    x, y = next(iter(train_loader))
    x = x.to(device)
    y = y.to(device)

    loss = model(x, y)
    print("smoke test loss:", loss.item())

    functional.reset_net(model)


    checkpoint_dir = Path(__file__).resolve().parent.parent / "results" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)



    for epoch in range(trainer_config.max_epochs):
        model.train()
        total_loss = 0.0
        total_batches = 0

        for step, (x, y) in enumerate(train_loader):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            loss = model(x, y)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                trainer_config.grad_norm_clip
            )

            optimizer.step()
            functional.reset_net(model)

            total_loss += loss.item()
            total_batches += 1

            if step % trainer_config.log_every == 0:
                print(f"epoch {epoch+1} step {step} loss {loss.item():.4f}")

        train_loss = total_loss / max(total_batches, 1)
        val_loss = evaluate(model, val_loader, device)

        print(f"epoch {epoch+1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        ckpt_path = checkpoint_dir / f"epoch_{epoch+1}.pt"
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
            },
            ckpt_path,
        )


if __name__ == "__main__":
    main()

