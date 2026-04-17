from pathlib import Path
import math
import torch
from torch.utils.data import DataLoader
from spikingjelly.activation_based import functional

from dataset import Enwik8Dataset
from config import TrainerConfig, get_sanity_model_config
from model import SpikingGPT

MAX_TEST_STEPS = 25
E_MAC_PJ = 4.5
E_AC_PJ = 0.9

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
            if total_batches >= MAX_TEST_STEPS:
                break

    return total_loss / max(total_batches, 1)


def print_energy_table_values(T, d, r_hat):
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

    print("\nEnergy Table Inputs")
    print(f"T={T}")
    print(f"d={d}")
    print(f"R_hat={r_hat:.6f}")
    print(f"E_MAC={E_MAC_PJ} pJ")
    print(f"E_AC={E_AC_PJ} pJ")

    print("\nEnergy Table Values")
    print(
        f"Q/R,K,V: vanilla = E_MAC * 3*T*d^2 = {E_MAC_PJ} * 3 * {T} * {d}^2 = {vanilla_qkv:.4e} pJ"
    )
    print(
        f"Q/R,K,V: spikegpt = E_AC * R_hat * 3*T*d^2 = {E_AC_PJ} * {r_hat:.6f} * 3 * {T} * {d}^2 = {spike_qkv:.4e} pJ"
    )
    print(
        f"f(Q/R,K,V): vanilla = E_MAC * 2*T^2*d = {E_MAC_PJ} * 2 * {T}^2 * {d} = {vanilla_attn:.4e} pJ"
    )
    print(
        f"f(Q/R,K,V): spikegpt = E_MAC * 7*T*d = {E_MAC_PJ} * 7 * {T} * {d} = {spike_attn:.4e} pJ"
    )
    print(
        f"Scale: vanilla = E_MAC * T^2 = {E_MAC_PJ} * {T}^2 = {vanilla_scale:.4e} pJ"
    )
    print(
        f"Softmax: vanilla = E_MAC * 2*T^2 = {E_MAC_PJ} * 2 * {T}^2 = {vanilla_softmax:.4e} pJ"
    )
    print(
        f"FFN layer 1: vanilla = E_MAC * FL_MLP1 = {E_MAC_PJ} * ({d} * {4*d}) = {vanilla_ffn1:.4e} pJ"
    )
    print(
        f"FFN layer 1: spikegpt = E_AC * R_hat * FL_MLP1 = {E_AC_PJ} * {r_hat:.6f} * ({d} * {4*d}) = {spike_ffn1:.4e} pJ"
    )
    print(
        f"FFN layer 2: vanilla = E_MAC * FL_MLP2 = {E_MAC_PJ} * ({d} * {d}) = {vanilla_ffn2:.4e} pJ"
    )
    print(
        f"FFN layer 2: spikegpt = E_AC * R_hat * FL_MLP2 = {E_AC_PJ} * {r_hat:.6f} * ({d} * {d}) = {spike_ffn2:.4e} pJ"
    )
    print(
        f"FFN layer 3: vanilla = E_MAC * FL_MLP3 = {E_MAC_PJ} * ({4*d} * {d}) = {vanilla_ffn3:.4e} pJ"
    )
    print(
        f"FFN layer 3: spikegpt = E_AC * R_hat * FL_MLP3 = {E_AC_PJ} * {r_hat:.6f} * ({4*d} * {d}) = {spike_ffn3:.4e} pJ"
    )
    print(f"Overall vanilla = {vanilla_overall:.4e} pJ")
    print(f"Overall spikegpt = {spike_overall:.4e} pJ")
    print(f"Overall ratio V/S = {vanilla_overall / max(spike_overall, 1e-12):.4f}x")


def print_table_row(method, model_config, train_bpc, test_bpc, params_m):
    print("\nTable-ready row")
    print(f"Method: {method}")
    print("Spiking: yes")
    print(f"L: {model_config.n_layer}")
    print(f"d: {model_config.n_embd}")
    print(f"T: {model_config.ctx_len}")
    print(f"Train BPC: {train_bpc:.4f}")
    print(f"Test BPC: {test_bpc:.4f}")
    print("Complexity: O(T·d)")
    print(f"Params (M): {params_m:.4f}")


def main():
    model_config = get_sanity_model_config()
    trainer_config = TrainerConfig(batch_size=4)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    data_dir = Path(__file__).resolve().parent.parent / "data" / "enwik8_split"
    checkpoint_dir = Path(__file__).resolve().parent.parent / "results" / "checkpoints"
    checkpoint_files = sorted(checkpoint_dir.glob("*.pt"))

    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")

    print("available checkpoints:")
    for p in checkpoint_files:
        print(" -", p.name)

    ckpt_path = checkpoint_files[-1]
    print("using checkpoint:", ckpt_path.name)

    train_dataset = Enwik8Dataset(data_dir / "train.txt", model_config.ctx_len)
    train_loader = DataLoader(
        train_dataset,
        batch_size=trainer_config.batch_size,
        shuffle=False,
        num_workers=trainer_config.num_workers,
    )

    test_dataset = Enwik8Dataset(data_dir / "test.txt", model_config.ctx_len)
    test_loader = DataLoader(
        test_dataset,
        batch_size=trainer_config.batch_size,
        shuffle=False,
        num_workers=trainer_config.num_workers,
    )

    model = SpikingGPT(model_config).to(device)
    model.reset_spike_stats()


    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    num_params = sum(p.numel() for p in model.parameters())
    params_m = num_params / 1e6

    model.reset_spike_stats()
    train_loss = evaluate(model, train_loader, device)
    train_bpc = train_loss / math.log(2)

    model.reset_spike_stats()
    test_loss = evaluate(model, test_loader, device)
    test_ppl = torch.exp(torch.tensor(test_loss))
    test_bpc = test_loss / math.log(2)

    spike_nonzero, spike_total = model.get_spike_stats()
    r_hat = spike_nonzero / max(spike_total, 1)


    print(f"test_loss={test_loss:.4f}")
    print(f"test_perplexity={test_ppl.item():.4f}")
    print(f"test_bpc={test_bpc:.4f}")
    print(f"R_hat={r_hat:.6f}")
    print(f"params_m={params_m:.4f}")
    print_energy_table_values(model_config.ctx_len, model_config.n_embd, r_hat)
    print_table_row(
        method="SpikeGPT small reimplementation",
        model_config=model_config,
        train_bpc=train_bpc,
        test_bpc=test_bpc,
        params_m=params_m,
    )



if __name__ == "__main__":
    main()
