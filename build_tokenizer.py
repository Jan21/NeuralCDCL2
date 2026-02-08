"""
One-time script to build the whitespace tokenizer from training data.

Usage:
    python build_tokenizer.py [--train_path output/train.json] [--save_path models/tokenizer.json]
"""

import argparse
from pathlib import Path

from neural_cdcl.tokenizer import build_tokenizer, load_tokenizer, get_vocab_size


def main():
    parser = argparse.ArgumentParser(description="Build tokenizer from training data")
    parser.add_argument(
        "--train_path",
        type=str,
        default="output/train.json",
        help="Path to training JSON file",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="tokenizer/tokenizer.json",
        help="Path to save the tokenizer",
    )
    args = parser.parse_args()

    print(f"Building tokenizer from {args.train_path}...")
    tokenizer = build_tokenizer(args.train_path, args.save_path)
    print(f"Tokenizer saved to {args.save_path}")
    print(f"Vocabulary size: {get_vocab_size(tokenizer)}")

    # Verify roundtrip
    tok = load_tokenizer(args.save_path)
    test_text = "SOLVE_BEGIN CALL unit_propagate"
    encoded = tok.encode(test_text)
    decoded = tok.decode(encoded.ids)
    print(f"Roundtrip test: '{test_text}' -> {encoded.ids} -> '{decoded}'")


if __name__ == "__main__":
    main()
