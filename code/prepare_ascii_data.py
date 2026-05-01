"""
Download and prepare an ASCII art dataset for fine-tuning.

Default source: HuggingFace  jncraton/ascii-art-world
  Each row has an "art" column with one ASCII drawing.

Alternative: pass --input to process a local text file where pieces are
  separated by blank lines or --- / === dividers.

Output: a single UTF-8 text file where pieces are joined by
  a '========================================' separator line.

Usage:
  python prepare_ascii_data.py
  python prepare_ascii_data.py --dataset jncraton/ascii-art-world
  python prepare_ascii_data.py --input my_collection.txt
  python prepare_ascii_data.py --input my_collection.txt --output data/ascii_art/train.txt
"""
import argparse
import os
import re
import sys
from pathlib import Path


SEPARATOR = '\n' + '=' * 40 + '\n'


def load_from_huggingface(dataset_name):
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' library not found. Run: pip install datasets")
        sys.exit(1)

    print(f"Loading '{dataset_name}' from HuggingFace...")
    ds = load_dataset(dataset_name)

    pieces = []
    for split_name, split_data in ds.items():
        text_col = None
        for col in ['content', 'art', 'text', 'ascii', 'drawing']:
            if col in split_data.column_names:
                text_col = col
                break
        if text_col is None:
            text_col = split_data.column_names[0]

        print(f"  {split_name}: {len(split_data):,} examples (column '{text_col}')")
        for row in split_data:
            val = str(row[text_col]).strip()
            if val:
                pieces.append(val)

    return pieces


def load_from_file(input_path):
    text = Path(input_path).read_text(encoding='utf-8', errors='replace')
    pieces = re.split(r'\n{3,}|\n---+\n|\n===+\n', text)
    pieces = [p.strip() for p in pieces if p.strip() and len(p.strip()) > 5]
    return pieces


def save_dataset(pieces, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = SEPARATOR.join(pieces)
    output_path.write_text(content, encoding='utf-8')
    size_kb = len(content.encode()) / 1024
    print(f"Saved {len(pieces):,} pieces → {output_path}  ({size_kb:.1f} KB)")


def main():
    parser = argparse.ArgumentParser(description='Prepare ASCII art data for fine-tuning')
    parser.add_argument('--dataset', default='AvaLovelace/ASCII-Art',
                        help='HuggingFace dataset name (default: AvaLovelace/ASCII-Art — '
                             '5.3k human-drawn pieces from asciiart.eu)')
    parser.add_argument('--input', default=None,
                        help='Local text file with ASCII art pieces (skips HuggingFace)')
    parser.add_argument('--output', default=None,
                        help='Output file path (default: data/ascii_art/train.txt)')
    args = parser.parse_args()

    # Resolve output path relative to project root
    project_root = Path(__file__).resolve().parent.parent
    output_path = Path(args.output) if args.output else project_root / 'data' / 'ascii_art' / 'train.txt'

    if args.input:
        print(f"Loading from local file: {args.input}")
        pieces = load_from_file(args.input)
    else:
        pieces = load_from_huggingface(args.dataset)

    if not pieces:
        print("ERROR: No data loaded. Check your dataset name or input file.")
        sys.exit(1)

    print(f"Total pieces: {len(pieces):,}")
    save_dataset(pieces, output_path)


if __name__ == '__main__':
    main()
