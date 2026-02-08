"""
Metrics for CDCL trace prediction.

- Token accuracy: % of non-masked tokens predicted correctly
- Exact match accuracy: full sequence match
- Per-command exact match: accuracy per command type
"""

from typing import Dict, List, Tuple

import torch

from .trace_parser import compute_per_command_accuracy


def token_accuracy(
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[int, int]:
    """
    Compute token-level accuracy on masked positions.

    Args:
        preds: (batch, seq_len) predicted token IDs
        targets: (batch, seq_len) ground truth token IDs
        mask: (batch, seq_len) loss mask (1 = compute, 0 = skip)

    Returns:
        (n_correct, n_total)
    """
    correct = ((preds == targets) & (mask == 1)).sum().item()
    total = (mask == 1).sum().item()
    return correct, total


def exact_match(
    preds: torch.Tensor,
    targets: torch.Tensor,
    mask: torch.Tensor,
) -> Tuple[int, int]:
    """
    Compute exact match accuracy (all non-masked tokens must match).

    Args:
        preds: (batch, seq_len) predicted token IDs
        targets: (batch, seq_len) ground truth token IDs
        mask: (batch, seq_len) loss mask (1 = compute, 0 = skip)

    Returns:
        (n_exact_match, n_total_sequences)
    """
    # A sequence is exact match if all masked positions are correct
    token_correct = (preds == targets) | (mask == 0)
    seq_correct = token_correct.all(dim=-1)
    return seq_correct.sum().item(), preds.size(0)


def per_command_metrics(
    pred_texts: List[str],
    gt_texts: List[str],
    loss_masks: List[List[int]],
) -> Dict[str, Tuple[int, int]]:
    """
    Compute per-command exact match accuracy across a batch.

    Args:
        pred_texts: List of predicted trace strings.
        gt_texts: List of ground truth trace strings.
        loss_masks: List of loss mask lists.

    Returns:
        Dict mapping command name -> (n_correct, n_total) aggregated.
    """
    aggregated: Dict[str, Tuple[int, int]] = {}

    for pred_text, gt_text, mask in zip(pred_texts, gt_texts, loss_masks):
        pred_tokens = pred_text.split()
        gt_tokens = gt_text.split()

        results = compute_per_command_accuracy(pred_tokens, gt_tokens, mask)

        for cmd, (correct, total) in results.items():
            if cmd not in aggregated:
                aggregated[cmd] = (0, 0)
            prev_correct, prev_total = aggregated[cmd]
            aggregated[cmd] = (prev_correct + correct, prev_total + total)

    return aggregated
