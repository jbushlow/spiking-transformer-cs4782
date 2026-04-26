import argparse
import random
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from spikingjelly.activation_based import functional

from dataset import Enwik8Dataset
from config import TrainerConfig, get_sanity_model_config
from model import SpikingGPT
from utils.checkpoint import find_latest_checkpoint, load_checkpoint, save_checkpoint

MAX_TRAIN_STEPS_PER_EPOCH = 200
MAX_VAL_STEPS = 50


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--resume",
        choices=["latest", "none"],
        default="latest",
        help="Resume from the newest checkpoint or start from scratch.",
    )
    parser.add_argument(
        "--resume-path",
        default=None,
        help="Resume from a specific checkpoint path instead of using --resume latest.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default=None,
        help="Directory where checkpoints are saved and searched.",
    )
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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

            if total_batches >= MAX_VAL_STEPS:
                break

    return total_loss / max(total_batches, 1)


def build_train_loader(dataset, trainer_config, epoch_index):
    generator = torch.Generator()
    generator.manual_seed(trainer_config.seed + epoch_index)
    return DataLoader(
        dataset,
        batch_size=trainer_config.batch_size,
        shuffle=True,
        num_workers=trainer_config.num_workers,
        generator=generator,
    )


def get_checkpoint_to_resume(args, trainer_config):
    if args.resume_path:
        return Path(args.resume_path)

    if args.resume == "latest" and trainer_config.auto_resume:
        return find_latest_checkpoint(trainer_config.epoch_save_path)

    return None


def restore_training_state(checkpoint, model, optimizer, device):
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if "rng_state_torch" in checkpoint:
        torch.set_rng_state(checkpoint["rng_state_torch"])
    if (
        device == "cuda"
        and "rng_state_cuda" in checkpoint
        and checkpoint["rng_state_cuda"] is not None
    ):
        torch.cuda.set_rng_state_all(checkpoint["rng_state_cuda"])

    start_epoch = checkpoint["epoch"]
    global_step = checkpoint.get("global_step", 0)

    return start_epoch, global_step


def main():
    args = parse_args()

    model_config = get_sanity_model_config()
    trainer_config = TrainerConfig(
        max_epochs=1,
        batch_size=4,
    )

    if args.checkpoint_dir is not None:
        trainer_config.epoch_save_path = args.checkpoint_dir
    if args.resume_path is not None:
        trainer_config.resume_path = args.resume_path
    trainer_config.auto_resume = args.resume == "latest"

    set_seed(trainer_config.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("using device:", device)

    data_dir = Path(__file__).resolve().parent.parent / "data" / "enwik8_split"

    train_dataset = Enwik8Dataset(data_dir / "train.txt", model_config.ctx_len)
    val_dataset = Enwik8Dataset(data_dir / "valid.txt", model_config.ctx_len)

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

    train_loader = build_train_loader(train_dataset, trainer_config, 0)
    x, y = next(iter(train_loader))
    x = x.to(device)
    y = y.to(device)

    loss = model(x, y)
    print("smoke test loss:", loss.item())
    functional.reset_net(model)

    checkpoint_dir = Path(trainer_config.epoch_save_path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = get_checkpoint_to_resume(args, trainer_config)
    start_epoch = 0
    global_step = 0

    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint = load_checkpoint(checkpoint_path, device)
        start_epoch, global_step = restore_training_state(checkpoint, model, optimizer, device)
        print(
            f"resuming from {checkpoint_path} "
            f"(starting at epoch {start_epoch + 1})"
        )
    else:
        print("starting training from scratch")

    for epoch in range(start_epoch, trainer_config.max_epochs):
        model.train()
        train_loader = build_train_loader(train_dataset, trainer_config, epoch)
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
                trainer_config.grad_norm_clip,
            )

            optimizer.step()
            functional.reset_net(model)

            total_loss += loss.item()
            total_batches += 1
            global_step += 1

            if step % trainer_config.log_every == 0:
                print(f"epoch {epoch + 1} step {step} loss {loss.item():.4f}")

            if total_batches % trainer_config.step_checkpoint_every == 0:
                ckpt_path = checkpoint_dir / f"epoch_{epoch + 1}_step_{total_batches}.pt"
                save_checkpoint(
                    ckpt_path,
                    {
                        "epoch": epoch + 1,
                        "step_in_epoch": total_batches,
                        "global_step": global_step,
                        "completed_epoch": False,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_loss": total_loss / max(total_batches, 1),
                        "loss_sum": total_loss,
                        "rng_state_torch": torch.get_rng_state(),
                        "rng_state_cuda": torch.cuda.get_rng_state_all() if device == "cuda" else None,
                        "model_config": vars(model_config),
                        "trainer_config": vars(trainer_config),
                        "config": vars(model_config),
                    },
                )

            if total_batches >= MAX_TRAIN_STEPS_PER_EPOCH:
                break

        train_loss = total_loss / max(total_batches, 1)
        val_loss = evaluate(model, val_loader, device)

        print(f"epoch {epoch + 1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")

        ckpt_path = checkpoint_dir / f"epoch_{epoch + 1}.pt"
        save_checkpoint(
            ckpt_path,
            {
                "epoch": epoch + 1,
                "global_step": global_step,
                "completed_epoch": True,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "loss_sum": total_loss,
                "rng_state_torch": torch.get_rng_state(),
                "rng_state_cuda": torch.cuda.get_rng_state_all() if device == "cuda" else None,
                "model_config": vars(model_config),
                "trainer_config": vars(trainer_config),
                "config": vars(model_config),
            },
        )


if __name__ == "__main__":
    main()
