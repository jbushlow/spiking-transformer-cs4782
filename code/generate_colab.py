from pathlib import Path
import re

import torch
import torch.nn.functional as F
from spikingjelly.activation_based import functional

from config import get_sanity_model_config
from model import SpikingGPT


DEFAULT_PROMPT = "The "
DEFAULT_MAX_NEW_TOKENS = 200
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_K = 20

CHECKPOINT_PATTERN = re.compile(r"^epoch_(\d+)(?:_step_(\d+))?\.pt$")


def encode(text: str) -> list[int]:
    return list(text.encode("utf-8"))


def decode(tokens: list[int]) -> str:
    return bytes(tokens).decode("utf-8", errors="replace")


def checkpoint_sort_key(path: Path):
    match = CHECKPOINT_PATTERN.match(path.name)
    if match is None:
        return (-1, -1, path.name)

    epoch = int(match.group(1))
    is_epoch_end = 1 if match.group(2) is None else 0
    step = int(match.group(2) or 0)
    return (epoch, is_epoch_end, step, path.name)


def get_latest_checkpoint() -> Path:
    checkpoint_dir = Path(__file__).resolve().parent.parent / "results" / "checkpoints"
    checkpoint_files = sorted(checkpoint_dir.glob("*.pt"), key=checkpoint_sort_key)

    if not checkpoint_files:
        raise FileNotFoundError(f"No checkpoint files found in {checkpoint_dir}")

    return checkpoint_files[-1]


def load_model(device: str) -> SpikingGPT:
    checkpoint_path = get_latest_checkpoint()
    config = get_sanity_model_config()

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = SpikingGPT(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print(f"using device: {device}")
    print(f"using checkpoint: {checkpoint_path.name}")
    print(f"ctx_len={config.ctx_len} n_embd={config.n_embd} n_layer={config.n_layer}")
    return model


@torch.no_grad()
def generate_text(
    model: SpikingGPT,
    prompt: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_k: int = DEFAULT_TOP_K,
    device: str = "cpu",
) -> str:
    tokens = encode(prompt)

    for _ in range(max_new_tokens):
        context = tokens[-model.ctx_len:]
        idx = torch.tensor([context], dtype=torch.long, device=device)

        logits = model(idx)
        functional.reset_net(model)

        next_logits = logits[0, -1, :]
        if temperature != 1.0:
            next_logits = next_logits / temperature

        if top_k > 0:
            k = min(top_k, next_logits.size(-1))
            top_values, _ = torch.topk(next_logits, k)
            cutoff = top_values[-1]
            next_logits = next_logits.masked_fill(next_logits < cutoff, float("-inf"))

        probs = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()
        tokens.append(next_token)

    return decode(tokens)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(device)
    generated = generate_text(model, DEFAULT_PROMPT, device=device)

    print("\nprompt:")
    print(repr(DEFAULT_PROMPT))
    print("\ngenerated:")
    print(generated)


if __name__ == "__main__":
    main()
