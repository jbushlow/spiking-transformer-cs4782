import argparse
import csv
import math
import string
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from spikingjelly.activation_based import functional

from config import GPTConfig, get_sanity_model_config
from model import SpikingGPT
from utils.checkpoint import find_latest_checkpoint, load_checkpoint


PUNCTUATION_BYTES = {ord(ch) for ch in string.punctuation}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a checkpoint. Defaults to latest in results/checkpoints.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/spike_heatmaps",
        help="Directory where heatmaps and summaries are written.",
    )
    parser.add_argument(
        "--prompt",
        action="append",
        default=[],
        help="Prompt to analyze. Repeat this flag to compare multiple prompts.",
    )
    parser.add_argument(
        "--layers",
        default="all",
        help="Comma-separated zero-based layer ids to plot, or 'all'.",
    )
    parser.add_argument(
        "--modules",
        choices=["rwkv", "rffn", "both"],
        default="both",
        help="Which spike-producing modules to visualize.",
    )
    parser.add_argument(
        "--max-neurons",
        type=int,
        default=96,
        help="Maximum number of neurons to plot per heatmap.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Heatmap image resolution.",
    )
    return parser.parse_args()


def find_checkpoint(explicit_path):
    if explicit_path is not None:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    checkpoint_dir = Path(__file__).resolve().parent.parent / "results" / "checkpoints"
    checkpoint_path = find_latest_checkpoint(checkpoint_dir)
    if checkpoint_path is None:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")
    return checkpoint_path


def build_model_config(checkpoint):
    model_config_dict = checkpoint.get("model_config") or checkpoint.get("config")
    if model_config_dict:
        return GPTConfig(**model_config_dict)
    return get_sanity_model_config()


def parse_layers(spec, n_layer):
    if spec == "all":
        return list(range(n_layer))

    layers = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        layer_id = int(part)
        if layer_id < 0 or layer_id >= n_layer:
            raise ValueError(f"Layer id {layer_id} is out of range 0..{n_layer - 1}")
        layers.append(layer_id)
    if not layers:
        raise ValueError("No valid layers were provided.")
    return sorted(set(layers))


def slugify_prompt(prompt, index):
    cleaned = []
    for ch in prompt[:40]:
        if ch.isalnum():
            cleaned.append(ch.lower())
        elif ch in {" ", "-", "_"}:
            cleaned.append("_")
    suffix = "".join(cleaned).strip("_") or f"prompt_{index + 1}"
    return f"{index + 1:02d}_{suffix}"


def encode_prompt(prompt, ctx_len):
    prompt_bytes = prompt.encode("utf-8")
    if not prompt_bytes:
        raise ValueError("Prompt cannot be empty.")
    if len(prompt_bytes) > ctx_len:
        raise ValueError(
            f"Prompt is {len(prompt_bytes)} bytes long, which exceeds ctx_len={ctx_len}."
        )
    tensor = torch.tensor(list(prompt_bytes), dtype=torch.long).unsqueeze(0)
    return tensor, list(prompt_bytes)


def printable_byte(byte_value):
    if 32 <= byte_value <= 126:
        return chr(byte_value)
    if byte_value == 10:
        return r"\n"
    if byte_value == 9:
        return r"\t"
    return "."


def format_token_labels(byte_values):
    return [printable_byte(b) for b in byte_values]


def collect_spikes(model, idx, device, layer_ids, modules):
    captures = {}
    hooks = []

    def save_output(name):
        def hook(_, __, output):
            captures[name] = output[0].detach().cpu()
        return hook

    module_names = []
    if modules in {"rwkv", "both"}:
        module_names.append("rwkv")
    if modules in {"rffn", "both"}:
        module_names.append("rffn")

    for layer_id in layer_ids:
        block = model.blocks[layer_id]
        if "rwkv" in module_names:
            hooks.append(
                block.spiking_rwkv.register_forward_hook(
                    save_output(f"layer_{layer_id:02d}_rwkv")
                )
            )
        if "rffn" in module_names:
            hooks.append(
                block.spiking_rffn.register_forward_hook(
                    save_output(f"layer_{layer_id:02d}_rffn")
                )
            )

    model.eval()
    with torch.no_grad():
        _ = model(idx.to(device))
    functional.reset_net(model)

    for hook in hooks:
        hook.remove()

    return captures


def save_heatmap(spikes, byte_values, title, output_path, max_neurons, dpi):
    time_steps, total_neurons = spikes.shape
    neurons_to_plot = min(max_neurons, total_neurons)
    data = spikes[:, :neurons_to_plot].transpose(0, 1).numpy()
    labels = format_token_labels(byte_values)

    fig_width = max(8.0, time_steps * 0.35)
    fig_height = max(4.5, neurons_to_plot * 0.06)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(data, aspect="auto", interpolation="nearest", cmap="hot", vmin=0.0, vmax=1.0)
    ax.set_title(title)
    ax.set_xlabel("Prompt byte / timestep")
    ax.set_ylabel(f"Neuron index (first {neurons_to_plot})")
    ax.set_xticks(range(time_steps))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    fig.colorbar(image, ax=ax, label="Spike")
    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def save_prompt_overview(prompt, byte_values, output_path):
    lines = [
        f"prompt: {prompt}",
        f"bytes: {len(byte_values)}",
        "timestep,byte,display,is_punctuation",
    ]
    for index, byte_value in enumerate(byte_values):
        lines.append(
            f"{index},{byte_value},{printable_byte(byte_value)},{int(byte_value in PUNCTUATION_BYTES)}"
        )
    output_path.write_text("\n".join(lines) + "\n")


def spike_rate_for_positions(spikes, positions):
    if not positions:
        return math.nan
    subset = spikes[positions]
    return float(subset.mean().item())


def summarize_prompt(prompt, byte_values, captures):
    punctuation_positions = [i for i, byte_value in enumerate(byte_values) if byte_value in PUNCTUATION_BYTES]
    non_punctuation_positions = [i for i, byte_value in enumerate(byte_values) if byte_value not in PUNCTUATION_BYTES]
    rows = []

    for name, spikes in sorted(captures.items()):
        layer_text, module_name = name.rsplit("_", 1)
        layer_id = int(layer_text.split("_")[1])
        overall_rate = float(spikes.mean().item())
        punctuation_rate = spike_rate_for_positions(spikes, punctuation_positions)
        non_punctuation_rate = spike_rate_for_positions(spikes, non_punctuation_positions)

        rows.append(
            {
                "prompt": prompt,
                "layer": layer_id,
                "module": module_name,
                "timesteps": spikes.shape[0],
                "neurons": spikes.shape[1],
                "spike_rate": overall_rate,
                "punctuation_spike_rate": punctuation_rate,
                "non_punctuation_spike_rate": non_punctuation_rate,
                "punctuation_count": len(punctuation_positions),
            }
        )

    return rows


def write_summary_csv(rows, output_path):
    fieldnames = [
        "prompt",
        "layer",
        "module",
        "timesteps",
        "neurons",
        "spike_rate",
        "punctuation_spike_rate",
        "non_punctuation_spike_rate",
        "punctuation_count",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    prompts = args.prompt or [
        "Hello, world!",
        "Hello world",
        "Why do spikes matter in language models?",
    ]

    checkpoint_path = find_checkpoint(args.checkpoint)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = load_checkpoint(checkpoint_path, device)
    model_config = build_model_config(checkpoint)
    layer_ids = parse_layers(args.layers, model_config.n_layer)

    model = SpikingGPT(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    for prompt_index, prompt in enumerate(prompts):
        idx, byte_values = encode_prompt(prompt, model_config.ctx_len)
        captures = collect_spikes(model, idx, device, layer_ids, args.modules)

        prompt_slug = slugify_prompt(prompt, prompt_index)
        prompt_dir = output_dir / prompt_slug
        prompt_dir.mkdir(parents=True, exist_ok=True)
        save_prompt_overview(prompt, byte_values, prompt_dir / "prompt.txt")

        for name, spikes in sorted(captures.items()):
            layer_text, module_name = name.rsplit("_", 1)
            layer_id = int(layer_text.split("_")[1])
            title = f"Layer {layer_id} {module_name.upper()} | {prompt}"
            output_path = prompt_dir / f"{name}.png"
            save_heatmap(
                spikes=spikes,
                byte_values=byte_values,
                title=title,
                output_path=output_path,
                max_neurons=args.max_neurons,
                dpi=args.dpi,
            )

        summary_rows.extend(summarize_prompt(prompt, byte_values, captures))

    write_summary_csv(summary_rows, output_dir / "summary.csv")
    print(f"saved spike heatmaps to {output_dir}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"prompts analyzed: {len(prompts)}")


if __name__ == "__main__":
    main()
