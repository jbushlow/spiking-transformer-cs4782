import sys
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from spikingjelly.activation_based import functional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import GPTConfig
from model import SpikingGPT


def encode(text: str) -> list:
    return list(text.encode("utf-8"))


def decode(tokens: list) -> str:
    return bytes(tokens).decode("utf-8", errors="replace")


def load_model(checkpoint_path: str, config: GPTConfig, device: str) -> SpikingGPT:
    ckpt = torch.load(checkpoint_path, map_location=device)
    model = SpikingGPT(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    epoch = ckpt.get("epoch", "?")
    val_loss = ckpt.get("val_loss", float("nan"))
    print(f"Loaded checkpoint  epoch={epoch}  val_loss={val_loss:.4f}")
    return model


@torch.no_grad()
def generate(
    model: SpikingGPT,
    prompt_ids: list,
    max_new_tokens: int,
    temperature: float = 1.0,
    top_k: int = 40,
    device: str = "cpu",
    stream_callback=None,
) -> list:
    model.eval()
    ctx_len = model.ctx_len
    tokens = list(prompt_ids)

    for _ in range(max_new_tokens):
        context = tokens[-ctx_len:]
        idx = torch.tensor([context], dtype=torch.long, device=device)

        logits = model(idx)          # (1, T, vocab_size)
        functional.reset_net(model)  # reset LIF state so it doesn't carry over between steps

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

        if stream_callback is not None:
            stream_callback(decode([next_token]))

    return tokens


def main():
    parser = argparse.ArgumentParser(description="Generate text with SpikingGPT")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to a .pt checkpoint saved by train.py")
    parser.add_argument("--prompt", type=str, default=" ",
                        help="Seed text for generation")
    parser.add_argument("--max_new_tokens", type=int, default=200,
                        help="Number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Sampling temperature (lower = sharper, higher = more random)")
    parser.add_argument("--top_k", type=int, default=40,
                        help="Top-k sampling cutoff (0 = disabled, use full distribution)")
    parser.add_argument("--greedy", action="store_true",
                        help="Use greedy (argmax) decoding instead of sampling")
    # these need to match whatever config you used when training
    parser.add_argument("--ctx_len", type=int, default=128)
    parser.add_argument("--n_embd",  type=int, default=128)
    parser.add_argument("--n_layer", type=int, default=2)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    config = GPTConfig(ctx_len=args.ctx_len, n_embd=args.n_embd, n_layer=args.n_layer)
    model = load_model(args.checkpoint, config, device)

    prompt_ids = encode(args.prompt)
    print(f"Prompt ({len(prompt_ids)} tokens): {repr(args.prompt)}")
    print("-" * 50)
    print(args.prompt, end="", flush=True)

    if args.greedy:
        model.eval()
        tokens = list(prompt_ids)
        with torch.no_grad():
            for _ in range(args.max_new_tokens):
                context = tokens[-model.ctx_len:]
                idx = torch.tensor([context], dtype=torch.long, device=device)
                logits = model(idx)
                functional.reset_net(model)
                next_token = logits[0, -1, :].argmax().item()
                tokens.append(next_token)
                print(decode([next_token]), end="", flush=True)
    else:
        generate(
            model, prompt_ids, args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device,
            stream_callback=lambda ch: print(ch, end="", flush=True),
        )

    print()


if __name__ == "__main__":
    main()
