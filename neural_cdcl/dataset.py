"""
Dataset and preprocessing for CDCL trace training.

Handles:
- Streaming JSON parsing for large training files
- Binary pre-processing for efficient training
- Memory-mapped dataset loading
- Collation with padding
- Test set loading
"""

import json
import logging
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from .masking import compute_loss_mask
from .tokenizer import EOS_TOKEN, PAD_ID, load_tokenizer, encode_tokens

logger = logging.getLogger(__name__)


def stream_json_texts(path: str):
    """
    Stream text entries from a JSON file of format [{"text": "..."}, ...].

    Parses line-by-line to avoid loading the entire file into memory.
    """
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith('"text"'):
                colon_idx = line.index(":")
                value = line[colon_idx + 1:].strip()
                if value.endswith(","):
                    value = value[:-1]
                text = json.loads(value)
                yield text


def preprocess_training_data(
    train_path: str,
    tokenizer_path: str,
    output_dir: str,
    max_seq_len: int,
):
    """
    Pre-tokenize and pre-mask the training data into binary files.

    Creates:
        - tokens.bin: flat uint16 array of token IDs
        - masks.bin: flat uint8 array of loss mask values
        - index.npy: (N, 2) int64 array of [offset, length]
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokens_path = output_dir / "tokens.bin"
    masks_path = output_dir / "masks.bin"
    index_path = output_dir / "index.npy"

    # Check if already preprocessed
    if tokens_path.exists() and masks_path.exists() and index_path.exists():
        logger.info("Pre-processed data already exists at %s, skipping.", output_dir)
        return

    tokenizer = load_tokenizer(tokenizer_path)
    eos_id = tokenizer.token_to_id(EOS_TOKEN)

    all_token_ids = []
    all_masks = []
    index_entries = []
    offset = 0
    total = 0
    skipped = 0

    logger.info("Pre-processing training data from %s...", train_path)

    for text in stream_json_texts(train_path):
        total += 1
        tokens = text.split()

        # +1 for EOS token
        if len(tokens) + 1 > max_seq_len:
            skipped += 1
            continue

        # Compute loss mask on raw tokens
        mask = compute_loss_mask(tokens)

        # Append EOS
        tokens.append(EOS_TOKEN)
        mask.append(1)  # EOS should be predicted

        # Encode via direct vocab lookup (avoids pre-tokenizer splitting special tokens)
        ids = encode_tokens(tokenizer, tokens)

        index_entries.append([offset, len(ids)])
        all_token_ids.extend(ids)
        all_masks.extend(mask)
        offset += len(ids)

        if total % 100000 == 0:
            logger.info("Processed %d entries (%d skipped)...", total, skipped)

    logger.info(
        "Done. Total: %d, Kept: %d, Skipped (too long): %d",
        total, total - skipped, skipped,
    )

    # Write binary files
    np.array(all_token_ids, dtype=np.uint16).tofile(str(tokens_path))
    np.array(all_masks, dtype=np.uint8).tofile(str(masks_path))
    np.array(index_entries, dtype=np.int64).reshape(-1, 2).tofile(str(index_path))

    logger.info("Saved preprocessed data to %s", output_dir)


class CDCLTrainDataset(Dataset):
    """Memory-mapped dataset for pre-processed training data."""

    def __init__(self, preprocessed_dir: str, max_seq_len: int):
        preprocessed_dir = Path(preprocessed_dir)

        # Load index
        index_data = np.fromfile(str(preprocessed_dir / "index.npy"), dtype=np.int64)
        self.index = index_data.reshape(-1, 2)

        # Memory-map token and mask arrays
        self.tokens = np.memmap(
            str(preprocessed_dir / "tokens.bin"), dtype=np.uint16, mode="r"
        )
        self.masks = np.memmap(
            str(preprocessed_dir / "masks.bin"), dtype=np.uint8, mode="r"
        )

        # Filter to samples within max_seq_len
        self.valid_indices = np.where(self.index[:, 1] <= max_seq_len)[0]

        logger.info(
            "Loaded %d training samples (%d total, %d filtered by length)",
            len(self.valid_indices),
            len(self.index),
            len(self.index) - len(self.valid_indices),
        )

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        real_idx = self.valid_indices[idx]
        offset, length = self.index[real_idx]
        token_ids = torch.tensor(
            self.tokens[offset : offset + length].copy(), dtype=torch.long
        )
        loss_mask = torch.tensor(
            self.masks[offset : offset + length].copy(), dtype=torch.long
        )
        return token_ids, loss_mask


class CDCLTestDataset(Dataset):
    """Dataset for test sets (small enough to fit in memory)."""

    def __init__(self, json_path: str, tokenizer_path: str, max_seq_len: int, max_samples: int = 0):
        tokenizer = load_tokenizer(tokenizer_path)
        eos_id = tokenizer.token_to_id(EOS_TOKEN)

        with open(json_path, "r") as f:
            data = json.load(f)

        self.samples = []
        self.raw_texts = []
        skipped = 0

        for entry in data:
            if max_samples > 0 and len(self.samples) >= max_samples:
                break

            text = entry["text"]
            tokens = text.split()

            if len(tokens) + 1 > max_seq_len:
                skipped += 1
                continue

            mask = compute_loss_mask(tokens)
            tokens.append(EOS_TOKEN)
            mask.append(1)

            ids = encode_tokens(tokenizer, tokens)

            self.samples.append((
                torch.tensor(ids, dtype=torch.long),
                torch.tensor(mask, dtype=torch.long),
            ))
            self.raw_texts.append(text)

        logger.info(
            "Loaded %s: %d samples (%d skipped, too long)",
            json_path, len(self.samples), skipped,
        )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]



def collate_fn(batch):
    """Pad sequences to the longest in the batch."""
    token_ids_list, loss_mask_list = zip(*batch)

    max_len = max(t.size(0) for t in token_ids_list)
    batch_size = len(batch)

    padded_tokens = torch.full((batch_size, max_len), PAD_ID, dtype=torch.long)
    padded_masks = torch.zeros((batch_size, max_len), dtype=torch.long)
    attention_masks = torch.zeros((batch_size, max_len), dtype=torch.bool)

    for i, (t, m) in enumerate(zip(token_ids_list, loss_mask_list)):
        seq_len = t.size(0)
        padded_tokens[i, :seq_len] = t
        padded_masks[i, :seq_len] = m
        attention_masks[i, :seq_len] = True

    return padded_tokens, padded_masks, attention_masks


def create_train_dataloader(
    preprocessed_dir: str,
    max_seq_len: int,
    batch_size: int,
    num_workers: int = 4,
) -> DataLoader:
    """Create training DataLoader with random sampling."""
    dataset = CDCLTrainDataset(preprocessed_dir, max_seq_len)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


def create_test_dataloader(
    json_path: str,
    tokenizer_path: str,
    max_seq_len: int,
    batch_size: int,
    num_workers: int = 2,
    max_samples: int = 0,
) -> DataLoader:
    """Create test DataLoader."""
    dataset = CDCLTestDataset(json_path, tokenizer_path, max_seq_len, max_samples=max_samples)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=num_workers,
        pin_memory=True,
    )
