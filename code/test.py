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
    model_config = GPTConfig(ctx_len=256, n_embd=128, n_layer=2)
    trainer_config = TrainerConfig(batch_size=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir = Path(__file__).resolve().parent.parent / "data" / "enwik8_split"
    ckpt_path = Path(__file__).resolve().parent.parent / "results" / "checkpoints" / "epoch_2.pt"

    test_dataset = Enwik8Dataset(data_dir / "test.txt", model_config.ctx_len)
    test_loader = DataLoader(
        test_dataset,
        batch_size=trainer_config.batch_size,
        shuffle=False,
        num_workers=trainer_config.num_workers,
    )

    model = SpikingGPT(model_config).to(device)

    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss = evaluate(model, test_loader, device)
    test_ppl = torch.exp(torch.tensor(test_loss))

    print(f"test_loss={test_loss:.4f}")
    print(f"test_perplexity={test_ppl.item():.4f}")


if __name__ == "__main__":
    main()
