import torch
from pathlib import Path
import argparse

from config import TrainerConfig, get_spikegpt_46m_config
from dataset import Enwik8Dataset
from model_learnable_beta import SpikingGPTLearnableBeta
from torch.utils.data import DataLoader
from spikingjelly.activation_based import functional

from train import (
    MAX_TRAIN_STEPS_PER_EPOCH,
    build_train_loader,
    compute_lr,
    estimate_tokens_seen_from_epoch,
    evaluate,
    get_checkpoint_to_resume,
    parse_args,
    prune_checkpoints,
    restore_training_state,
    save_checkpoint,
    set_seed,
)
from utils.checkpoint import load_checkpoint


DEFAULT_MAX_TRAIN_STEPS_PER_EPOCH = MAX_TRAIN_STEPS_PER_EPOCH


def parse_args_learnable_beta():
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
    parser.add_argument(
        "--planned-epochs-remaining",
        type=int,
        default=25,
        help="How many more epochs you expect to train from this run. Used to set the LR decay horizon.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Optional override for the starting learning rate used by the schedule.",
    )
    parser.add_argument(
        "--lr-final",
        type=float,
        default=None,
        help="Optional override for the final learning-rate floor.",
    )
    parser.add_argument(
        "--preserve-lr-schedule",
        action="store_true",
        help="When resuming, reuse learning-rate schedule parameters saved in the checkpoint.",
    )
    parser.add_argument(
        "--constant-lr",
        action="store_true",
        help="Disable cosine LR decay and keep the learning rate constant.",
    )
    parser.add_argument(
        "--max-train-steps-per-epoch",
        type=int,
        default=DEFAULT_MAX_TRAIN_STEPS_PER_EPOCH,
        help="Maximum number of training batches per epoch.",
    )
    parser.add_argument(
        "--log-every-steps",
        type=int,
        default=None,
        help="Optional override for training loss/beta logging frequency.",
    )
    return parser.parse_args()


def collect_beta_diagnostics(model):
    rwkv_means = []
    rffn_means = []
    all_beta_values = []

    for name, p in model.named_parameters():
        if "beta_raw" not in name:
            continue
        beta = torch.sigmoid(p.detach()).float().cpu()
        all_beta_values.append(beta.reshape(-1))
        beta_mean = beta.mean().item()
        if "spiking_rwkv" in name:
            rwkv_means.append(beta_mean)
        elif "spiking_rffn" in name:
            rffn_means.append(beta_mean)

    if not all_beta_values:
        return None

    all_betas = torch.cat(all_beta_values)
    return {
        "overall_mean": all_betas.mean().item(),
        "overall_min": all_betas.min().item(),
        "overall_max": all_betas.max().item(),
        "overall_std": all_betas.std(unbiased=False).item(),
        "max_abs_delta": torch.max(torch.abs(all_betas - 0.5)).item(),
        "rwkv_mean": sum(rwkv_means) / len(rwkv_means) if rwkv_means else None,
        "rffn_mean": sum(rffn_means) / len(rffn_means) if rffn_means else None,
    }


def print_beta_diagnostics(model, prefix="beta"):
    stats = collect_beta_diagnostics(model)
    if stats is None:
        return
    print(
        f"{prefix}: overall_mean={stats['overall_mean']:.6f} "
        f"rwkv_mean={stats['rwkv_mean']:.6f} "
        f"rffn_mean={stats['rffn_mean']:.6f} "
        f"min={stats['overall_min']:.6f} "
        f"max={stats['overall_max']:.6f} "
        f"std={stats['overall_std']:.6f} "
        f"max|beta-0.5|={stats['max_abs_delta']:.6f}"
    )


def restore_training_state_learnable_beta(checkpoint, model, optimizer, device):
    checkpoint_state = checkpoint["model_state_dict"]
    current_state = model.state_dict()
    compatible_state = {
        key: value
        for key, value in checkpoint_state.items()
        if key in current_state and current_state[key].shape == value.shape
    }
    missing_keys, unexpected_keys = model.load_state_dict(compatible_state, strict=False)
    if missing_keys:
        print(f"initialized new learnable-beta parameters: {missing_keys}")
    if unexpected_keys:
        print(f"ignored incompatible checkpoint parameters: {unexpected_keys}")

    optimizer_state = checkpoint.get("optimizer_state_dict")
    if optimizer_state is not None:
        try:
            optimizer.load_state_dict(optimizer_state)
        except Exception as exc:
            print(f"warning: could not restore optimizer state ({exc}); starting optimizer fresh")

    rng_state_torch = checkpoint.get("rng_state_torch")
    if rng_state_torch is not None:
        try:
            if not isinstance(rng_state_torch, torch.Tensor):
                rng_state_torch = torch.as_tensor(rng_state_torch, dtype=torch.uint8)
            else:
                rng_state_torch = rng_state_torch.to(dtype=torch.uint8, device="cpu")
            torch.set_rng_state(rng_state_torch)
        except Exception as exc:
            print(f"warning: could not restore torch RNG state ({exc})")

    rng_state_cuda = checkpoint.get("rng_state_cuda")
    if device == "cuda" and rng_state_cuda is not None:
        try:
            if isinstance(rng_state_cuda, (list, tuple)):
                cuda_states = []
                for state in rng_state_cuda:
                    if not isinstance(state, torch.Tensor):
                        state = torch.as_tensor(state, dtype=torch.uint8)
                    else:
                        state = state.to(dtype=torch.uint8, device="cpu")
                    cuda_states.append(state)
                torch.cuda.set_rng_state_all(cuda_states)
            else:
                if not isinstance(rng_state_cuda, torch.Tensor):
                    rng_state_cuda = torch.as_tensor(rng_state_cuda, dtype=torch.uint8)
                else:
                    rng_state_cuda = rng_state_cuda.to(dtype=torch.uint8, device="cpu")
                torch.cuda.set_rng_state(rng_state_cuda)
        except Exception as exc:
            print(f"warning: could not restore CUDA RNG state ({exc})")

    start_epoch = checkpoint["epoch"]
    global_step = checkpoint.get("global_step", 0)
    tokens_seen = checkpoint.get("tokens_seen")
    saved_trainer_config = checkpoint.get("trainer_config")
    return start_epoch, global_step, tokens_seen, saved_trainer_config


def infer_legacy_final_tokens(saved_trainer_config, model_config):
    max_epochs = saved_trainer_config.get("max_epochs")
    epoch_length_fixed = saved_trainer_config.get("epoch_length_fixed")
    if max_epochs is None or epoch_length_fixed is None:
        return None
    return int(max_epochs) * int(epoch_length_fixed) * model_config.ctx_len


def main():
    args = parse_args_learnable_beta()

    project_root = Path(__file__).resolve().parent.parent
    default_checkpoint_dir = project_root / "results" / "checkpoints_learnable_beta"
    model_config = get_spikegpt_46m_config()
    trainer_config = TrainerConfig(
        max_epochs=1000,
        batch_size=8,
        step_checkpoint_every=100,
        log_every=50,
        keep_last_epoch_checkpoints=2,
        keep_last_step_checkpoints=1,
        epoch_save_path=str(default_checkpoint_dir),
    )

    if args.checkpoint_dir is not None:
        trainer_config.epoch_save_path = args.checkpoint_dir
    if args.resume_path is not None:
        trainer_config.resume_path = args.resume_path
    trainer_config.auto_resume = args.resume == "latest"
    trainer_config.warmup_tokens = 0
    trainer_config.schedule_start_tokens = 0
    if args.constant_lr:
        trainer_config.lr_decay = False
    if args.log_every_steps is not None:
        trainer_config.log_every = args.log_every_steps

    set_seed(trainer_config.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("using device:", device)

    data_dir = project_root / "data" / "enwik8_split"
    train_dataset = Enwik8Dataset(data_dir / "train.txt", model_config.ctx_len)
    val_dataset = Enwik8Dataset(data_dir / "valid.txt", model_config.ctx_len)

    val_loader = DataLoader(
        val_dataset,
        batch_size=trainer_config.batch_size,
        shuffle=False,
        num_workers=trainer_config.num_workers,
    )

    model = SpikingGPTLearnableBeta(model_config).to(device)
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
    print_beta_diagnostics(model, prefix="initial beta")
    functional.reset_net(model)

    checkpoint_dir = Path(trainer_config.epoch_save_path)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"saving checkpoints to: {checkpoint_dir}")

    checkpoint_path = get_checkpoint_to_resume(args, trainer_config)
    start_epoch = 0
    global_step = 0
    tokens_seen = 0
    saved_trainer_config = None

    if checkpoint_path is not None and checkpoint_path.exists():
        checkpoint = load_checkpoint(checkpoint_path, device)
        start_epoch, global_step, tokens_seen, saved_trainer_config = restore_training_state_learnable_beta(
            checkpoint, model, optimizer, device
        )
        if tokens_seen is None:
            tokens_seen = estimate_tokens_seen_from_epoch(start_epoch, trainer_config, model_config)
            print(f"warning: checkpoint missing tokens_seen; estimated tokens_seen={tokens_seen}")
        print(f"resuming from {checkpoint_path} (starting at epoch {start_epoch + 1})")
    else:
        print("starting training from scratch")

    if args.preserve_lr_schedule and saved_trainer_config:
        saved_learning_rate = saved_trainer_config.get("learning_rate")
        saved_lr_final = saved_trainer_config.get("lr_final")
        saved_final_tokens = saved_trainer_config.get("final_tokens")
        saved_warmup_tokens = saved_trainer_config.get("warmup_tokens")
        saved_schedule_start_tokens = saved_trainer_config.get("schedule_start_tokens")
        saved_step_checkpoint_every = saved_trainer_config.get("step_checkpoint_every")

        if saved_learning_rate is not None:
            trainer_config.learning_rate = saved_learning_rate
        if saved_lr_final is not None:
            trainer_config.lr_final = saved_lr_final
        if saved_warmup_tokens is not None:
            trainer_config.warmup_tokens = saved_warmup_tokens
        if saved_schedule_start_tokens is not None:
            trainer_config.schedule_start_tokens = saved_schedule_start_tokens
        if saved_step_checkpoint_every is not None:
            trainer_config.step_checkpoint_every = saved_step_checkpoint_every
        if saved_final_tokens is not None:
            trainer_config.final_tokens = saved_final_tokens
            print(
                f"preserving LR schedule from checkpoint: learning_rate={trainer_config.learning_rate:.6e}, "
                f"lr_final={trainer_config.lr_final:.6e}, "
                f"schedule_start_tokens={trainer_config.schedule_start_tokens}, "
                f"final_tokens={trainer_config.final_tokens}"
            )
        else:
            inferred_final_tokens = infer_legacy_final_tokens(saved_trainer_config, model_config)
            if inferred_final_tokens is not None:
                trainer_config.final_tokens = inferred_final_tokens
                print(
                    f"preserving inferred legacy LR horizon: learning_rate={trainer_config.learning_rate:.6e}, "
                    f"lr_final={trainer_config.lr_final:.6e}, "
                    f"schedule_start_tokens={trainer_config.schedule_start_tokens}, "
                    f"final_tokens={trainer_config.final_tokens}"
                )

    if args.learning_rate is not None:
        trainer_config.learning_rate = args.learning_rate
    if args.lr_final is not None:
        trainer_config.lr_final = args.lr_final

    max_train_steps_per_epoch = args.max_train_steps_per_epoch
    tokens_per_epoch = max_train_steps_per_epoch * trainer_config.batch_size * model_config.ctx_len
    if not (
        args.preserve_lr_schedule
        and saved_trainer_config
        and getattr(trainer_config, "final_tokens", None) is not None
    ):
        trainer_config.schedule_start_tokens = tokens_seen
        trainer_config.final_tokens = tokens_seen + args.planned_epochs_remaining * tokens_per_epoch

    print(
        f"lr schedule params: learning_rate={trainer_config.learning_rate:.6e}, "
        f"lr_final={trainer_config.lr_final:.6e}"
    )
    print(
        f"lr schedule horizon: schedule_start_tokens={trainer_config.schedule_start_tokens}, "
        f"current tokens_seen={tokens_seen}, "
        f"planned_epochs_remaining={args.planned_epochs_remaining}, "
        f"final_tokens={trainer_config.final_tokens}"
    )

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

            torch.nn.utils.clip_grad_norm_(model.parameters(), trainer_config.grad_norm_clip)

            optimizer.step()
            tokens_seen += y.numel()
            lr = compute_lr(tokens_seen, trainer_config)
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr
            functional.reset_net(model)

            total_loss += loss.item()
            total_batches += 1
            global_step += 1

            if step % trainer_config.log_every == 0:
                print(f"epoch {epoch + 1} step {step} loss {loss.item():.4f} lr {lr:.6e}")
                print_beta_diagnostics(model, prefix=f"epoch {epoch + 1} step {step} beta")

            if (
                total_batches % trainer_config.step_checkpoint_every == 0
                and total_batches < max_train_steps_per_epoch
            ):
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
                        "tokens_seen": tokens_seen,
                        "train_loss": total_loss / max(total_batches, 1),
                        "loss_sum": total_loss,
                        "rng_state_torch": torch.get_rng_state(),
                        "rng_state_cuda": torch.cuda.get_rng_state_all() if device == "cuda" else None,
                        "model_config": vars(model_config),
                        "trainer_config": vars(trainer_config),
                        "config": vars(model_config),
                    },
                )
                prune_checkpoints(
                    checkpoint_dir,
                    keep_last_epoch_checkpoints=trainer_config.keep_last_epoch_checkpoints,
                    keep_last_step_checkpoints=trainer_config.keep_last_step_checkpoints,
                )

            if total_batches >= max_train_steps_per_epoch:
                break

        train_loss = total_loss / max(total_batches, 1)
        val_loss = evaluate(model, val_loader, device)

        print(f"epoch {epoch + 1}: train_loss={train_loss:.4f} val_loss={val_loss:.4f}")
        print_beta_diagnostics(model, prefix=f"epoch {epoch + 1} beta")

        ckpt_path = checkpoint_dir / f"epoch_{epoch + 1}.pt"
        save_checkpoint(
            ckpt_path,
            {
                "epoch": epoch + 1,
                "global_step": global_step,
                "completed_epoch": True,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "tokens_seen": tokens_seen,
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
        prune_checkpoints(
            checkpoint_dir,
            keep_last_epoch_checkpoints=trainer_config.keep_last_epoch_checkpoints,
            keep_last_step_checkpoints=trainer_config.keep_last_step_checkpoints,
        )


if __name__ == "__main__":
    main()
