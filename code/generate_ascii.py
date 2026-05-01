"""
Generate ASCII art with a fine-tuned (or base) SpikingGPT.

Usage:
  # After fine-tuning — let the model generate freely:
  python code/generate_ascii.py --checkpoint results/ascii_checkpoints/epoch_30.pt

  # Prompt with the first line of a drawing:
  python code/generate_ascii.py --checkpoint results/ascii_checkpoints/epoch_30.pt \\
      --prompt "  /\\_/\\ " --tokens 300 --temp 0.8

  # Try the base model (no fine-tuning) — useful for comparison:
  python code/generate_ascii.py --checkpoint epoch_185.pt --tokens 500

  # Generate multiple samples:
  python code/generate_ascii.py --checkpoint results/ascii_checkpoints/epoch_30.pt \\
      --n 5 --tokens 400 --temp 0.9
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from spikingjelly.activation_based import functional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from config import GPTConfig
from model import SpikingGPT


def encode(text: str) -> list:
    return list(text.encode('utf-8'))


def decode(tokens: list) -> str:
    return bytes(tokens).decode('utf-8', errors='replace')


def load_model(checkpoint_path: str, device: str) -> SpikingGPT:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    saved = ckpt.get('config', ckpt.get('model_config', {}))
    config = GPTConfig(**{k: v for k, v in saved.items() if k != 'n_ffn'})
    model = SpikingGPT(config).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    val_loss = ckpt.get('val_loss', float('nan'))
    print(f'Loaded  epoch={ckpt.get("epoch", "?")}  val_loss={val_loss:.4f}  '
          f'ctx_len={config.ctx_len}  n_embd={config.n_embd}  n_layer={config.n_layer}')
    return model


@torch.no_grad()
def generate(
    model: SpikingGPT,
    prompt_ids: list,
    max_new_tokens: int,
    temperature: float = 0.85,
    top_k: int = 40,
    device: str = 'cpu',
    stream: bool = True,
) -> list:
    model.eval()
    ctx_len = model.ctx_len
    tokens  = list(prompt_ids)

    for _ in range(max_new_tokens):
        context = tokens[-ctx_len:]
        idx     = torch.tensor([context], dtype=torch.long, device=device)

        logits = model(idx)             # (1, T, vocab_size)
        functional.reset_net(model)     # clear LIF state between steps

        next_logits = logits[0, -1, :]

        if temperature != 1.0:
            next_logits = next_logits / temperature

        if top_k > 0:
            k = min(top_k, next_logits.size(-1))
            top_vals, _ = torch.topk(next_logits, k)
            next_logits[next_logits < top_vals[-1]] = float('-inf')

        probs      = F.softmax(next_logits, dim=-1)
        next_token = torch.multinomial(probs, 1).item()
        tokens.append(next_token)

        if stream:
            print(decode([next_token]), end='', flush=True)

    return tokens


def main():
    parser = argparse.ArgumentParser(description='Generate ASCII art with SpikingGPT')
    parser.add_argument('--checkpoint', required=True, help='Path to .pt checkpoint')
    parser.add_argument('--prompt',  default='\n',
                        help='Seed text (default: newline).  '
                             'Try starting with the first line of a drawing.')
    parser.add_argument('--tokens',  type=int,   default=400,  help='Tokens to generate')
    parser.add_argument('--temp',    type=float, default=0.85,
                        help='Sampling temperature. Lower (0.6–0.8) = more structured, '
                             'higher (1.0+) = more creative/chaotic')
    parser.add_argument('--top-k',   type=int,   default=40,
                        help='Top-k cutoff (0 = full distribution)')
    parser.add_argument('--n',       type=int,   default=1,    help='Number of samples')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model  = load_model(args.checkpoint, device)

    prompt_ids = encode(args.prompt)
    divider    = '=' * 40

    for i in range(args.n):
        header = f'\n{divider}\n[{i+1}/{args.n}]\n{divider}' if args.n > 1 else f'\n{divider}'
        print(header)
        print(args.prompt, end='', flush=True)

        generate(
            model, prompt_ids, args.tokens,
            temperature=args.temp,
            top_k=args.top_k,
            device=device,
            stream=True,
        )
        print()


if __name__ == '__main__':
    main()
