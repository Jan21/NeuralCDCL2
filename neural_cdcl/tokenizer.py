"""
Tokenizer utilities for CDCL traces.

Uses HuggingFace tokenizers library with a whitespace WordLevel tokenizer.
"""

import json
from pathlib import Path
from typing import List

from tokenizers import Tokenizer, models, pre_tokenizers, trainers


PAD_TOKEN = "<PAD>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"

SPECIAL_TOKENS = [PAD_TOKEN, EOS_TOKEN, UNK_TOKEN]
PAD_ID = 0
EOS_ID = 1
UNK_ID = 2


def build_tokenizer(train_path: str, save_path: str) -> Tokenizer:
    """
    Build a whitespace WordLevel tokenizer from training data.

    Streams the JSON file to collect unique tokens without loading
    the full 5GB+ file into memory.
    """
    vocab = set()

    with open(train_path, "r") as f:
        # Stream line-by-line to find "text" fields
        for line in f:
            line = line.strip()
            if line.startswith('"text"'):
                # Extract the value after "text": "..."
                # Format: "text": "TOKEN1 TOKEN2 ..."
                colon_idx = line.index(":")
                value = line[colon_idx + 1:].strip()
                # Remove trailing comma if present
                if value.endswith(","):
                    value = value[:-1]
                # Remove surrounding quotes
                text = json.loads(value)
                tokens = text.split()
                vocab.update(tokens)

    # Build vocabulary dict: special tokens first, then sorted vocab
    vocab_dict = {}
    for i, tok in enumerate(SPECIAL_TOKENS):
        vocab_dict[tok] = i

    for i, tok in enumerate(sorted(vocab), start=len(SPECIAL_TOKENS)):
        if tok not in vocab_dict:
            vocab_dict[tok] = i

    tokenizer = Tokenizer(models.WordLevel(vocab=vocab_dict, unk_token=UNK_TOKEN))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # Save
    save_dir = Path(save_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)
    tokenizer.save(save_path)

    return tokenizer


def load_tokenizer(path: str) -> Tokenizer:
    """Load a pre-built tokenizer from file."""
    return Tokenizer.from_file(path)


def encode(tokenizer: Tokenizer, text: str) -> List[int]:
    """Encode text to token IDs using the tokenizer's encode method."""
    return tokenizer.encode(text).ids


def encode_tokens(tokenizer: Tokenizer, tokens: List[str]) -> List[int]:
    """
    Encode a list of pre-split tokens directly to IDs via vocabulary lookup.

    This avoids issues with the Whitespace pre-tokenizer splitting
    special tokens like <EOS> into multiple sub-tokens.
    """
    vocab = tokenizer.get_vocab()
    unk_id = vocab.get(UNK_TOKEN, 0)
    return [vocab.get(tok, unk_id) for tok in tokens]


def decode_tokens(tokenizer: Tokenizer, ids: List[int]) -> List[str]:
    """Decode token IDs to a list of token strings."""
    id_to_token = {v: k for k, v in tokenizer.get_vocab().items()}
    return [id_to_token.get(i, UNK_TOKEN) for i in ids]


def decode(tokenizer: Tokenizer, ids: List[int]) -> str:
    """Decode token IDs back to text (space-joined)."""
    tokens = decode_tokens(tokenizer, ids)
    # Filter out special tokens for clean text output
    return " ".join(t for t in tokens if t not in (PAD_TOKEN, EOS_TOKEN))


def get_vocab_size(tokenizer: Tokenizer) -> int:
    """Get the vocabulary size."""
    return tokenizer.get_vocab_size()
