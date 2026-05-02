import argparse
import math
import re
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from spikingjelly.activation_based import functional
from tqdm import tqdm

from dataset import Enwik8Dataset
from config import GPTConfig, TrainerConfig, get_sanity_model_config
from model import SpikingGPT

E_MAC_PJ = 4.5
E_AC_PJ = 0.9
PAPER_ENERGY_T = 3072
PAPER_ENERGY_D = 512
PAPER_ENERGY_R_HAT = 0.15

CHECKPOINT_PATTERN = re.compile(r"^epoch_(\d+)(?:_step_(\d+))?\.pt$")

TABLE2_BASELINES = [
    ("Transformer", "no", "12", "512", "1024", "0.977", "1.137", r"$\mathcal{O}(T^2 \cdot d)$", "43.0M"),
    ("Reformer", "no", "12", "512", "1024", "1.040", "1.195", r"$\mathcal{O}(T \log T \cdot d)$", "40.1M"),
    ("Synthesizer", "no", "12", "512", "1024", "0.994", "1.298", r"$\mathcal{O}(T \cdot d^2)$", "42.8M"),
    ("Linear Transformer", "no", "12", "512", "1024", "0.981", "1.207", r"$\mathcal{O}(T \cdot d^2)$", "43.0M"),
    ("Performer", "no", "12", "512", "1024", "1.002", "1.199", r"$\mathcal{O}(T \cdot d^2 \log d)$", "43.0M"),
    ("Stacked LSTM", "no", "7", "-", "-", "1.420", "1.670", r"$\mathcal{O}(T \cdot d^2)$", "-"),
    ("SHA-LSTM (no attention)", "no", "4", "1024", "1024", "-", "1.330", r"$\mathcal{O}(T \cdot d^2)$", "-"),
    ("SpikeGPT 46M (paper)", "yes", "12", "512", "1024", "1.113", "1.283", r"$\mathcal{O}(T \cdot d)$", "46.1M"),
]

TABLE2_SUMMARY_METHODS = {
    "Transformer",
    "Performer",
    "Stacked LSTM",
    "SHA-LSTM (no attention)",
    "SpikeGPT 46M (paper)",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=None, help="Path to a checkpoint. Defaults to latest full epoch checkpoint in results/checkpoints.")
    parser.add_argument("--output-dir", default="results/latex_tables", help="Where to write the LaTeX tables.")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-test-steps", type=int, default=None, help="Optional cap on evaluation batches. Default evaluates the full selected split.")
    parser.add_argument("--progress-every", type=int, default=10, help="Print evaluation progress every N batches.")
    parser.add_argument("--include-train-bpc", action="store_true", help="Also evaluate the training split. By default only test BPC is computed.")
    parser.add_argument("--stride", type=int, default=None, help="Stride in bytes between evaluation windows. Defaults to ctx_len for non-overlapping chunks.")
    parser.add_argument("--energy-only", action="store_true", help="Only generate the energy table using a short evaluation run to estimate spike activity.")
    parser.add_argument("--bpc-only", action="store_true", help="Only generate the Enwik8 BPC table and summary, skipping energy-table generation.")
    return parser.parse_args()


class StridedEnwik8Dataset(Dataset):
    def __init__(self, path, ctx_len, stride=None):
        with open(path, "rb") as handle:
            data = handle.read()
        self.data = torch.tensor(list(data), dtype=torch.long)
        self.ctx_len = ctx_len
        self.stride = stride if stride is not None else ctx_len
        if self.stride <= 0:
            raise ValueError("stride must be positive")

        max_start = len(self.data) - (self.ctx_len + 1)
        if max_start < 0:
            self.starts = []
        else:
            self.starts = list(range(0, max_start + 1, self.stride))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, index):
        start = self.starts[index]
        x = self.data[start:start + self.ctx_len]
        y = self.data[start + 1:start + self.ctx_len + 1]
        return x, y


def evaluate(model, loader, device, max_steps=None, progress_every=10, split_name="eval"):
    model.eval()
    total_loss = 0.0
    total_batches = 0
    total_target = min(len(loader), max_steps) if max_steps is not None else len(loader)

    with torch.no_grad():
        with tqdm(total=total_target, desc=f"{split_name:>5}", unit="batch") as pbar:
            for x, y in loader:
                x = x.to(device)
                y = y.to(device)

                loss = model(x, y)
                total_loss += loss.item()
                total_batches += 1
                functional.reset_net(model)
                pbar.update(1)
                if progress_every > 0 and total_batches % progress_every == 0:
                    avg_loss = total_loss / total_batches
                    pbar.set_postfix(avg_loss=f"{avg_loss:.4f}")
                if max_steps is not None and total_batches >= max_steps:
                    break

    return total_loss / max(total_batches, 1)


def checkpoint_sort_key(path: Path):
    match = CHECKPOINT_PATTERN.match(path.name)
    if match is None:
        return (-1, -1, path.name)

    epoch = int(match.group(1))
    is_epoch_end = 1 if match.group(2) is None else 0
    step = int(match.group(2) or 0)
    return (epoch, is_epoch_end, step, path.name)


def find_checkpoint(explicit_path):
    if explicit_path is not None:
        path = Path(explicit_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {path}")
        return path

    checkpoint_dir = Path(__file__).resolve().parent.parent / "results" / "checkpoints"
    checkpoint_files = sorted(checkpoint_dir.glob("*.pt"), key=checkpoint_sort_key)
    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")

    full_epoch_checkpoints = [
        path for path in checkpoint_files
        if CHECKPOINT_PATTERN.match(path.name) and CHECKPOINT_PATTERN.match(path.name).group(2) is None
    ]
    if full_epoch_checkpoints:
        return full_epoch_checkpoints[-1]
    return checkpoint_files[-1]


def build_model_config(checkpoint):
    model_config_dict = checkpoint.get("model_config") or checkpoint.get("config")
    if model_config_dict:
        return GPTConfig(**model_config_dict)
    return get_sanity_model_config()


def sci_latex(value):
    if value == 0:
        return "0"
    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / (10 ** exponent)
    return rf"{mantissa:.2f} \times 10^{{{exponent}}}"


def ratio_latex(numerator, denominator):
    if denominator == 0:
        return "-"
    return f"{numerator / denominator:.2f}x"


def bpc_to_loss_str(bpc_str):
    if bpc_str == "-":
        return "-"
    return f"{float(bpc_str) * math.log(2):.3f}"


def compute_energy_metrics(T, d, r_hat):
    fl_mlp1 = d * (4 * d)
    fl_mlp2 = d * d
    fl_mlp3 = (4 * d) * d

    vanilla_qkv = E_MAC_PJ * 3 * T * (d ** 2)
    spike_qkv = E_AC_PJ * r_hat * 3 * T * (d ** 2)

    vanilla_attn = E_MAC_PJ * 2 * (T ** 2) * d
    spike_attn = E_MAC_PJ * 7 * T * d

    vanilla_scale = E_MAC_PJ * (T ** 2)
    vanilla_softmax = E_MAC_PJ * 2 * (T ** 2)

    vanilla_ffn1 = E_MAC_PJ * fl_mlp1
    spike_ffn1 = E_AC_PJ * r_hat * fl_mlp1

    vanilla_ffn2 = E_MAC_PJ * fl_mlp2
    spike_ffn2 = E_AC_PJ * r_hat * fl_mlp2

    vanilla_ffn3 = E_MAC_PJ * fl_mlp3
    spike_ffn3 = E_AC_PJ * r_hat * fl_mlp3

    vanilla_overall = (
        vanilla_qkv
        + vanilla_attn
        + vanilla_scale
        + vanilla_softmax
        + vanilla_ffn1
        + vanilla_ffn2
        + vanilla_ffn3
    )
    spike_overall = spike_qkv + spike_attn + spike_ffn1 + spike_ffn2 + spike_ffn3

    return {
        "vanilla_qkv": vanilla_qkv,
        "spike_qkv": spike_qkv,
        "vanilla_attn": vanilla_attn,
        "spike_attn": spike_attn,
        "vanilla_scale": vanilla_scale,
        "vanilla_softmax": vanilla_softmax,
        "vanilla_ffn1": vanilla_ffn1,
        "spike_ffn1": spike_ffn1,
        "vanilla_ffn2": vanilla_ffn2,
        "spike_ffn2": spike_ffn2,
        "vanilla_ffn3": vanilla_ffn3,
        "spike_ffn3": spike_ffn3,
        "vanilla_overall": vanilla_overall,
        "spike_overall": spike_overall,
    }


def build_energy_table_latex(model_config, r_hat):
    T = model_config.ctx_len
    d = model_config.n_embd
    m = compute_energy_metrics(T, d, r_hat)

    return rf"""\begin{{table*}}[t]
\centering
\caption{{Energy Evaluation. $F L_{{MLP}}$ represents the floating-point operations (FLOPs) of the MLP layers in the ANNs, while $\hat{{R}}$ denotes the spike firing rate. For this checkpoint, operations are parameterized with $T={T}$, $d={d}$, and $\hat{{R}}={r_hat:.4f}$. Energy assumes $E_{{MAC}}={E_MAC_PJ}$pJ and $E_{{AC}}={E_AC_PJ}$pJ.}}
\begin{{tabular}}{{llccccc}}
\hline
 &  & Vanilla GPT & SpikeGPT & \multicolumn{{2}}{{c}}{{Energy Consumption (pJ)}} & Ratio \\
\cline{{5-6}}
Module & Operation & Formula & Formula & Vanilla GPT & SpikeGPT & V/S \\
\hline
Attention & $Q/R, K, V$ & $E_{{MAC}} \cdot 3Td^2$ & $E_{{AC}} \cdot \hat{{R}} \cdot 3Td^2$ & ${sci_latex(m['vanilla_qkv'])}$ & ${sci_latex(m['spike_qkv'])}$ & {ratio_latex(m['vanilla_qkv'], m['spike_qkv'])} \\
Attention & $f(Q/R,K,V)$ & $E_{{MAC}} \cdot 2T^2d$ & $E_{{MAC}} \cdot 7Td$ & ${sci_latex(m['vanilla_attn'])}$ & ${sci_latex(m['spike_attn'])}$ & {ratio_latex(m['vanilla_attn'], m['spike_attn'])} \\
Attention & Scale & $E_{{MAC}} \cdot T^2$ & - & ${sci_latex(m['vanilla_scale'])}$ & - & - \\
Attention & Softmax & $E_{{MAC}} \cdot 2T^2$ & - & ${sci_latex(m['vanilla_softmax'])}$ & - & - \\
\hline
FFN & Layer 1 & $E_{{MAC}} \cdot FL_{{MLP1}}$ & $E_{{AC}} \cdot \hat{{R}} \cdot FL_{{MLP1}}$ & ${sci_latex(m['vanilla_ffn1'])}$ & ${sci_latex(m['spike_ffn1'])}$ & {ratio_latex(m['vanilla_ffn1'], m['spike_ffn1'])} \\
FFN & Layer 2 & $E_{{MAC}} \cdot FL_{{MLP2}}$ & $E_{{AC}} \cdot \hat{{R}} \cdot FL_{{MLP2}}$ & ${sci_latex(m['vanilla_ffn2'])}$ & ${sci_latex(m['spike_ffn2'])}$ & {ratio_latex(m['vanilla_ffn2'], m['spike_ffn2'])} \\
FFN & Layer 3 & $E_{{MAC}} \cdot FL_{{MLP3}}$ & $E_{{AC}} \cdot \hat{{R}} \cdot FL_{{MLP3}}$ & ${sci_latex(m['vanilla_ffn3'])}$ & ${sci_latex(m['spike_ffn3'])}$ & {ratio_latex(m['vanilla_ffn3'], m['spike_ffn3'])} \\
\hline
Overall & - & - & - & ${sci_latex(m['vanilla_overall'])}$ & ${sci_latex(m['spike_overall'])}$ & {ratio_latex(m['vanilla_overall'], m['spike_overall'])} \\
\hline
\end{{tabular}}
\end{{table*}}
"""


def build_energy_summary_table_latex(model_config, r_hat):
    paper_metrics = compute_energy_metrics(PAPER_ENERGY_T, PAPER_ENERGY_D, PAPER_ENERGY_R_HAT)
    ours_metrics = compute_energy_metrics(model_config.ctx_len, model_config.n_embd, r_hat)

    rows = [
        (
            "SpikeGPT 46M (paper)",
            str(PAPER_ENERGY_T),
            str(PAPER_ENERGY_D),
            f"{PAPER_ENERGY_R_HAT:.2f}",
            sci_latex(paper_metrics["vanilla_overall"]),
            sci_latex(paper_metrics["spike_overall"]),
            ratio_latex(paper_metrics["vanilla_overall"], paper_metrics["spike_overall"]),
        ),
        (
            "SpikeGPT 46M (ours)",
            str(model_config.ctx_len),
            str(model_config.n_embd),
            f"{r_hat:.4f}",
            sci_latex(ours_metrics["vanilla_overall"]),
            sci_latex(ours_metrics["spike_overall"]),
            ratio_latex(ours_metrics["vanilla_overall"], ours_metrics["spike_overall"]),
        ),
    ]

    body = "\n".join(
        f"{model_name} & {T} & {d} & {rhat} & ${vanilla}$ & ${spike}$ & {ratio} \\\\"
        for model_name, T, d, rhat, vanilla, spike, ratio in rows
    )

    return rf"""\begin{{table*}}[t]
\centering
\caption{{Simplified energy comparison for SpikeGPT 46M. The paper row uses the published energy-table setting $T={PAPER_ENERGY_T}$, $d={PAPER_ENERGY_D}$, and $\hat{{R}}={PAPER_ENERGY_R_HAT:.2f}$. The 'ours' row uses the measured checkpoint setting $T={model_config.ctx_len}$, $d={model_config.n_embd}$, and $\hat{{R}}={r_hat:.4f}$. Lower SpikeGPT energy and higher V/S ratio indicate stronger efficiency relative to the dense baseline.}}
\small
\setlength{{\tabcolsep}}{{4pt}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lcccccc}}
\hline
Model & $T$ & $d$ & $\hat{{R}}$ & Vanilla GPT Overall (pJ) & SpikeGPT Overall (pJ) & V/S Ratio \\
\hline
{body}
\hline
\end{{tabular}}%
}}
\end{{table*}}
"""


def build_bpc_table_latex(model_config, train_bpc, test_bpc, params_m):
    train_bpc_str = "-" if train_bpc is None else f"{train_bpc:.3f}"
    spike_row = (
        "SpikeGPT 46M (ours)",
        "yes",
        str(model_config.n_layer),
        str(model_config.n_embd),
        str(model_config.ctx_len),
        train_bpc_str,
        f"{test_bpc:.3f}",
        r"$\mathcal{O}(T \cdot d)$",
        f"{params_m:.1f}M",
    )

    rows = TABLE2_BASELINES + [spike_row]
    body = "\n".join(
        f"{method} & {spiking} & {L} & {d} & {T} & {train} & {test} & {complexity} & {params} \\\\"
        for method, spiking, L, d, T, train, test, complexity, params in rows
    )

    return rf"""\begin{{table*}}[t]
\centering
\caption{{Enwik8 results measured in bits per character (BPC). Lower is better. The SpikeGPT row below is generated from the evaluated checkpoint.}}
\small
\setlength{{\tabcolsep}}{{4pt}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lcccccccc}}
\hline
Method & Spiking & $L$ & $d$ & $T$ & Train BPC & Test BPC & Complexity & Params. \\
\hline
{body}
\hline
\end{{tabular}}%
}}
\end{{table*}}
"""


def build_loss_bpc_table_latex(model_config, train_loss, train_bpc, test_loss, test_bpc, params_m):
    train_loss_str = "-" if train_loss is None else f"{train_loss:.3f}"
    train_bpc_str = "-" if train_bpc is None else f"{train_bpc:.3f}"
    spike_row = (
        "SpikeGPT 46M (ours)",
        "yes",
        str(model_config.n_layer),
        str(model_config.n_embd),
        str(model_config.ctx_len),
        train_loss_str,
        train_bpc_str,
        f"{test_loss:.3f}",
        f"{test_bpc:.3f}",
        r"$\mathcal{O}(T \cdot d)$",
        f"{params_m:.1f}M",
    )

    baseline_rows = [
        (
            method,
            spiking,
            L,
            d,
            T,
            bpc_to_loss_str(train_bpc_str),
            train_bpc_str,
            bpc_to_loss_str(test_bpc_str),
            test_bpc_str,
            complexity,
            params,
        )
        for method, spiking, L, d, T, train_bpc_str, test_bpc_str, complexity, params in TABLE2_BASELINES
    ]

    rows = baseline_rows + [spike_row]
    body = "\n".join(
        f"{method} & {spiking} & {L} & {d} & {T} & {train_loss_val} & {train_bpc_val} & {test_loss_val} & {test_bpc_val} & {complexity} & {params} \\\\"
        for method, spiking, L, d, T, train_loss_val, train_bpc_val, test_loss_val, test_bpc_val, complexity, params in rows
    )

    return rf"""\begin{{table*}}[t]
\centering
\caption{{Enwik8 results reported in both cross-entropy loss and bits per character (BPC). Lower is better. Loss values are in nats/character; BPC is obtained by dividing loss by $\ln 2$. The SpikeGPT row below is generated from the evaluated checkpoint.}}
\small
\setlength{{\tabcolsep}}{{3pt}}
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{lcccccccccc}}
\hline
Method & Spiking & $L$ & $d$ & $T$ & Train Loss & Train BPC & Test Loss & Test BPC & Complexity & Params. \\
\hline
{body}
\hline
\end{{tabular}}%
}}
\end{{table*}}
"""


def build_bpc_summary_table_latex(model_config, train_bpc, test_bpc):
    spike_row = (
        "SpikeGPT 46M (ours)",
        "-" if train_bpc is None else f"{train_bpc:.3f}",
        f"{test_bpc:.3f}",
        r"$\mathcal{O}(T \cdot d)$",
    )

    baseline_rows = [
        (
            method,
            train_bpc_str,
            test_bpc_str,
            complexity,
        )
        for method, _spiking, _L, _d, _T, train_bpc_str, test_bpc_str, complexity, _params in TABLE2_BASELINES
        if method in TABLE2_SUMMARY_METHODS
    ]

    rows = baseline_rows + [spike_row]
    body = "\n".join(
        f"{method} & {train_bpc_val} & {test_bpc_val} & {complexity} \\\\"
        for method, train_bpc_val, test_bpc_val, complexity in rows
    )

    return rf"""\begin{{table*}}[t]
\centering
\caption{{Compact Enwik8 summary reported in bits per character (BPC). Lower is better. The SpikeGPT row below is generated from the evaluated checkpoint.}}
\small
\setlength{{\tabcolsep}}{{5pt}}
\resizebox{{0.8\textwidth}}{{!}}{{%
\begin{{tabular}}{{lccc}}
\hline
Method & Train BPC & Test BPC & Complexity \\
\hline
{body}
\hline
\end{{tabular}}%
}}
\end{{table*}}
"""


def main():
    args = parse_args()
    if args.energy_only and args.bpc_only:
        raise ValueError("--energy-only and --bpc-only cannot be used together.")

    checkpoint_path = find_checkpoint(args.checkpoint)
    print("using checkpoint:", checkpoint_path.name)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model_config = build_model_config(checkpoint)
    trainer_config = TrainerConfig(batch_size=args.batch_size)
    eval_stride = args.stride if args.stride is not None else model_config.ctx_len

    data_dir = Path(__file__).resolve().parent.parent / "data" / "enwik8_split"
    test_dataset = StridedEnwik8Dataset(data_dir / "test.txt", model_config.ctx_len, stride=eval_stride)
    test_loader = DataLoader(
        test_dataset,
        batch_size=trainer_config.batch_size,
        shuffle=False,
        num_workers=trainer_config.num_workers,
    )

    model = SpikingGPT(model_config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])

    num_params = sum(p.numel() for p in model.parameters())
    params_m = num_params / 1e6

    train_loss = None
    train_bpc = None
    if args.include_train_bpc and not args.energy_only:
        train_dataset = StridedEnwik8Dataset(data_dir / "train.txt", model_config.ctx_len, stride=eval_stride)
        train_loader = DataLoader(
            train_dataset,
            batch_size=trainer_config.batch_size,
            shuffle=False,
            num_workers=trainer_config.num_workers,
        )
        model.reset_spike_stats()
        train_loss = evaluate(
            model,
            train_loader,
            device,
            max_steps=args.max_test_steps,
            progress_every=args.progress_every,
            split_name="train",
        )
        train_bpc = train_loss / math.log(2)

    eval_max_steps = args.max_test_steps
    if args.energy_only and eval_max_steps is None:
        eval_max_steps = 1

    model.reset_spike_stats()
    test_bpc = None
    if args.energy_only:
        _ = evaluate(
            model,
            test_loader,
            device,
            max_steps=eval_max_steps,
            progress_every=args.progress_every,
            split_name="energy",
        )
    else:
        test_loss = evaluate(
            model,
            test_loader,
            device,
            max_steps=eval_max_steps,
            progress_every=args.progress_every,
            split_name="test",
        )
        test_bpc = test_loss / math.log(2)

    spike_nonzero, spike_total = model.get_spike_stats()
    r_hat = spike_nonzero / max(spike_total, 1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    energy_table_path = output_dir / "table1_energy.tex"
    energy_summary_table_path = output_dir / "table1_energy_summary.tex"
    bpc_table_path = output_dir / "table2_enwik8.tex"
    loss_bpc_table_path = output_dir / "table2_enwik8_loss_bpc.tex"
    loss_bpc_summary_table_path = output_dir / "table2_enwik8_summary.tex"
    summary_path = output_dir / "metrics_summary.txt"

    if not args.bpc_only:
        energy_table_path.write_text(build_energy_table_latex(model_config, r_hat))
        energy_summary_table_path.write_text(build_energy_summary_table_latex(model_config, r_hat))
    if not args.energy_only:
        bpc_table_path.write_text(build_bpc_table_latex(model_config, train_bpc, test_bpc, params_m))
        loss_bpc_table_path.write_text(
            build_loss_bpc_table_latex(model_config, train_loss, train_bpc, test_loss, test_bpc, params_m)
        )
        loss_bpc_summary_table_path.write_text(
            build_bpc_summary_table_latex(model_config, train_bpc, test_bpc)
        )
    summary_path.write_text(
        "\n".join(
            [
                f"checkpoint={checkpoint_path}",
                f"ctx_len={model_config.ctx_len}",
                f"n_embd={model_config.n_embd}",
                f"n_layer={model_config.n_layer}",
                f"train_loss={'not_computed' if train_loss is None else f'{train_loss:.6f}'}",
                f"train_bpc={'not_computed' if train_bpc is None else f'{train_bpc:.6f}'}",
                f"test_loss={'not_computed' if test_bpc is None else f'{test_loss:.6f}'}",
                f"test_bpc={'not_computed' if test_bpc is None else f'{test_bpc:.6f}'}",
                f"r_hat={r_hat:.6f}",
                f"params_m={params_m:.6f}",
                f"max_test_steps={'full' if eval_max_steps is None else eval_max_steps}",
                f"include_train_bpc={args.include_train_bpc}",
                f"stride={eval_stride}",
                f"energy_only={args.energy_only}",
            ]
        )
        + "\n"
    )

    if train_bpc is not None:
        print(f"train_bpc={train_bpc:.4f}")
    elif not args.energy_only:
        print("train_bpc=not computed")
    if test_bpc is not None:
        print(f"test_bpc={test_bpc:.4f}")
    elif args.energy_only:
        print("test_bpc=skipped (energy-only mode)")
    print(f"R_hat={r_hat:.6f}")
    print(f"params_m={params_m:.4f}")
    if not args.bpc_only:
        print(f"wrote {energy_table_path}")
        print(f"wrote {energy_summary_table_path}")
    if not args.energy_only:
        print(f"wrote {bpc_table_path}")
        print(f"wrote {loss_bpc_table_path}")
        print(f"wrote {loss_bpc_summary_table_path}")
    print(f"wrote {summary_path}")


if __name__ == "__main__":
    main()
