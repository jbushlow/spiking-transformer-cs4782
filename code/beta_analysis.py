import argparse
import csv
import re
from pathlib import Path

import matplotlib.pyplot as plt
import torch


CHECKPOINT_PATTERN = re.compile(r"^epoch_(\d+)\.pt$")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-dir",
        default="results/checkpoints_learnable_beta",
        help="Directory containing learnable-beta epoch checkpoints.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/beta_analysis",
        help="Directory to write plots and CSV summaries.",
    )
    return parser.parse_args()


def epoch_from_path(path: Path):
    match = CHECKPOINT_PATTERN.match(path.name)
    if match is None:
        return None
    return int(match.group(1))


def collect_beta_stats(model_state_dict):
    rwkv_tensors = []
    rffn_tensors = []
    all_tensors = []
    rwkv_layer_means = {}
    rffn_layer_means = {}

    for name, value in model_state_dict.items():
        if "beta_raw" not in name:
            continue
        beta = torch.sigmoid(value.detach().float().cpu()).reshape(-1)
        all_tensors.append(beta)

        layer_match = re.search(r"blocks\.(\d+)\.", name)
        layer_id = int(layer_match.group(1)) if layer_match else -1

        if "spiking_rwkv" in name:
            rwkv_tensors.append(beta)
            rwkv_layer_means[layer_id] = beta.mean().item()
        elif "spiking_rffn" in name:
            rffn_tensors.append(beta)
            rffn_layer_means[layer_id] = beta.mean().item()

    if not all_tensors:
        return None

    all_betas = torch.cat(all_tensors)
    rwkv_betas = torch.cat(rwkv_tensors) if rwkv_tensors else None
    rffn_betas = torch.cat(rffn_tensors) if rffn_tensors else None

    return {
        "overall_mean": all_betas.mean().item(),
        "overall_min": all_betas.min().item(),
        "overall_max": all_betas.max().item(),
        "overall_std": all_betas.std(unbiased=False).item(),
        "max_abs_delta": torch.max(torch.abs(all_betas - 0.5)).item(),
        "rwkv_mean": rwkv_betas.mean().item() if rwkv_betas is not None else None,
        "rffn_mean": rffn_betas.mean().item() if rffn_betas is not None else None,
        "rwkv_layer_means": rwkv_layer_means,
        "rffn_layer_means": rffn_layer_means,
    }


def load_epoch_stats(checkpoint_dir: Path):
    rows = []
    all_rwkv_layers = set()
    all_rffn_layers = set()

    for path in sorted(checkpoint_dir.glob("epoch_*.pt"), key=lambda p: epoch_from_path(p) or -1):
        epoch = epoch_from_path(path)
        if epoch is None:
            continue
        checkpoint = torch.load(path, map_location="cpu")
        stats = collect_beta_stats(checkpoint["model_state_dict"])
        if stats is None:
            continue

        row = {
            "epoch": epoch,
            "overall_mean": stats["overall_mean"],
            "overall_min": stats["overall_min"],
            "overall_max": stats["overall_max"],
            "overall_std": stats["overall_std"],
            "max_abs_delta": stats["max_abs_delta"],
            "rwkv_mean": stats["rwkv_mean"],
            "rffn_mean": stats["rffn_mean"],
        }
        for layer_id, mean_val in stats["rwkv_layer_means"].items():
            key = f"rwkv_layer_{layer_id}"
            row[key] = mean_val
            all_rwkv_layers.add(layer_id)
        for layer_id, mean_val in stats["rffn_layer_means"].items():
            key = f"rffn_layer_{layer_id}"
            row[key] = mean_val
            all_rffn_layers.add(layer_id)

        rows.append(row)

    return rows, sorted(all_rwkv_layers), sorted(all_rffn_layers)


def write_csv(rows, rwkv_layers, rffn_layers, output_path: Path):
    fieldnames = [
        "epoch",
        "overall_mean",
        "overall_min",
        "overall_max",
        "overall_std",
        "max_abs_delta",
        "rwkv_mean",
        "rffn_mean",
    ]
    fieldnames.extend(f"rwkv_layer_{layer}" for layer in rwkv_layers)
    fieldnames.extend(f"rffn_layer_{layer}" for layer in rffn_layers)

    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_overall_means(rows, output_path: Path):
    epochs = [row["epoch"] for row in rows]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["overall_mean"] for row in rows], label="Overall")
    plt.plot(epochs, [row["rwkv_mean"] for row in rows], label="RWKV")
    plt.plot(epochs, [row["rffn_mean"] for row in rows], label="RFFN")
    plt.axhline(0.5, color="black", linestyle="--", linewidth=1, label="Init 0.5")
    plt.xlabel("Epoch")
    plt.ylabel("Mean beta")
    plt.title("Mean Learned Beta by Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_beta_spread(rows, output_path: Path):
    epochs = [row["epoch"] for row in rows]
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, [row["overall_min"] for row in rows], label="Min")
    plt.plot(epochs, [row["overall_max"] for row in rows], label="Max")
    plt.plot(epochs, [row["overall_std"] for row in rows], label="Std")
    plt.plot(epochs, [row["max_abs_delta"] for row in rows], label="Max |beta-0.5|")
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Beta Spread by Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_layer_means(rows, layer_ids, prefix, output_path: Path):
    epochs = [row["epoch"] for row in rows]
    plt.figure(figsize=(9, 5))
    for layer_id in layer_ids:
        key = f"{prefix}_layer_{layer_id}"
        values = [row.get(key) for row in rows]
        plt.plot(epochs, values, label=f"Layer {layer_id}")
    plt.xlabel("Epoch")
    plt.ylabel("Mean beta")
    plt.title(f"{prefix.upper()} Layer Beta Means by Epoch")
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main():
    args = parse_args()
    checkpoint_dir = Path(args.checkpoint_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, rwkv_layers, rffn_layers = load_epoch_stats(checkpoint_dir)
    if not rows:
        raise FileNotFoundError(f"No learnable-beta epoch checkpoints found in {checkpoint_dir}")

    write_csv(rows, rwkv_layers, rffn_layers, output_dir / "beta_summary.csv")
    plot_overall_means(rows, output_dir / "beta_means.png")
    plot_beta_spread(rows, output_dir / "beta_spread.png")
    if rwkv_layers:
        plot_layer_means(rows, rwkv_layers, "rwkv", output_dir / "beta_rwkv_layers.png")
    if rffn_layers:
        plot_layer_means(rows, rffn_layers, "rffn", output_dir / "beta_rffn_layers.png")

    print(f"wrote beta analysis to {output_dir}")


if __name__ == "__main__":
    main()
