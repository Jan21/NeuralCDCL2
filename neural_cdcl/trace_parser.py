"""
Trace parser for fine-grained per-command metrics.

Segments a CDCL trace into command blocks, each representing
a single operation (e.g., EVALUATE_CLAUSE, QUEUE, WRITE_ASSIGNMENTS).
This enables computing exact match accuracy per command type.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

# Command sets duplicated from cdcl_data_gen.parser to avoid heavy __init__ imports.
READ_COMMANDS = {
    "READ_ASSIGNMENTS",
    "READ_CLAUSES",
    "READ_DECISION_LEVELS",
    "READ_LEVEL",
    "READ_CONFLICT_CLAUSE",
    "READ_REASON_CLAUSES",
    "READ_LEARNED_CLAUSE",
}

WRITE_COMMANDS = {
    "WRITE_ASSIGNMENTS",
    "WRITE_CONFLICT_CLAUSE",
    "WRITE_LEARNED_CLAUSE",
}

END_COMMANDS = {
    "UNIT_PROPAGATION_END",
    "ANALYZE_CONFLICT_END",
    "SOLVE_END",
    "SOLVE_ITER_END",
}

BEGIN_COMMANDS = {
    "SOLVE_BEGIN",
    "UNIT_PROPAGATION_BEGIN",
    "ANALYZE_CONFLICT_BEGIN",
}

# Commands followed by [ ... ] content
BRACKET_COMMANDS = {
    "QUEUE",
    "CHECK_VAR",
    "LATEST_ASSIGNED",
    "REASON",
    "CURRENT_LEVEL_VARS",
    "CURRENT_LEVEL_LITS",
    "LEARNED_LIT",
    "BRANCHING_VARIABLE",
    "N_ASSIGNED",
    "AT_LEVEL",
} | WRITE_COMMANDS

# Commands followed by { ... } content
BRACE_COMMANDS = {"LEVELS"}

# Commands with inline content up to next recognized command
INLINE_COMMANDS = {"EVALUATE_CLAUSE", "BACKTRACK", "CALL"}

# Standalone commands (no following content)
STANDALONE_COMMANDS = {
    "PROPAGATE",
    "NO_CONFLICT",
    "NO_PROPAGATION",
    "UIP",
    "SAT",
    "UNSAT",
    "LEVEL_UP",
} | BEGIN_COMMANDS | END_COMMANDS

# All recognized command tokens
ALL_COMMANDS = (
    BRACKET_COMMANDS
    | BRACE_COMMANDS
    | INLINE_COMMANDS
    | STANDALONE_COMMANDS
    | READ_COMMANDS
)


@dataclass
class TraceSegment:
    """A segment of a trace corresponding to one command."""

    command: str  # e.g., "QUEUE", "EVALUATE_CLAUSE"
    start_idx: int  # Start position in the token list
    end_idx: int  # End position (exclusive) in the token list
    tokens: List[str]  # All tokens in this segment


def segment_trace(tokens: List[str]) -> List[TraceSegment]:
    """
    Parse a trace into segments, each representing one command and its content.

    Args:
        tokens: Whitespace-split token list.

    Returns:
        List of TraceSegment objects.
    """
    segments = []
    n = len(tokens)
    i = 0

    while i < n:
        token = tokens[i]

        if token in READ_COMMANDS:
            # READ command + [ ... ] bracket content
            start = i
            j = i + 1
            if j < n and tokens[j] == "[":
                depth = 0
                while j < n:
                    if tokens[j] == "[":
                        depth += 1
                    elif tokens[j] == "]":
                        depth -= 1
                    j += 1
                    if depth == 0:
                        break
            segments.append(TraceSegment(
                command=token,
                start_idx=start,
                end_idx=j,
                tokens=tokens[start:j],
            ))
            i = j

        elif token in BRACKET_COMMANDS:
            # Command + [ ... ] bracket content
            start = i
            j = i + 1
            if j < n and tokens[j] == "[":
                depth = 0
                while j < n:
                    if tokens[j] == "[":
                        depth += 1
                    elif tokens[j] == "]":
                        depth -= 1
                    j += 1
                    if depth == 0:
                        break
            segments.append(TraceSegment(
                command=token,
                start_idx=start,
                end_idx=j,
                tokens=tokens[start:j],
            ))
            i = j

        elif token in BRACE_COMMANDS:
            # Command + { ... } brace content
            start = i
            j = i + 1
            if j < n and tokens[j] == "{":
                depth = 0
                while j < n:
                    if tokens[j] == "{":
                        depth += 1
                    elif tokens[j] == "}":
                        depth -= 1
                    j += 1
                    if depth == 0:
                        break
            segments.append(TraceSegment(
                command=token,
                start_idx=start,
                end_idx=j,
                tokens=tokens[start:j],
            ))
            i = j

        elif token in INLINE_COMMANDS:
            # Command + content up to next recognized command
            start = i
            j = i + 1
            while j < n and tokens[j] not in ALL_COMMANDS:
                j += 1
            segments.append(TraceSegment(
                command=token,
                start_idx=start,
                end_idx=j,
                tokens=tokens[start:j],
            ))
            i = j

        elif token in STANDALONE_COMMANDS:
            segments.append(TraceSegment(
                command=token,
                start_idx=i,
                end_idx=i + 1,
                tokens=[token],
            ))
            i += 1

        else:
            # Unknown token — skip
            i += 1

    return segments


def compute_per_command_accuracy(
    pred_tokens: List[str],
    gt_tokens: List[str],
    loss_mask: List[int],
) -> Dict[str, Tuple[int, int]]:
    """
    Compute per-command exact match accuracy.

    Uses ground truth segmentation and compares predicted tokens
    at the same positions (only for non-masked positions).

    Args:
        pred_tokens: Predicted token list.
        gt_tokens: Ground truth token list.
        loss_mask: Loss mask (1 = predicted, 0 = masked/given).

    Returns:
        Dict mapping command name -> (n_correct, n_total).
        A command instance is "correct" if all its non-masked tokens match.
    """
    segments = segment_trace(gt_tokens)
    results: Dict[str, Tuple[int, int]] = {}

    for seg in segments:
        cmd = seg.command
        if cmd not in results:
            results[cmd] = (0, 0)

        # Check if all non-masked tokens in this segment match
        correct = True
        has_predicted_tokens = False

        for pos in range(seg.start_idx, seg.end_idx):
            if pos >= len(loss_mask) or loss_mask[pos] == 0:
                continue  # Masked position, skip
            has_predicted_tokens = True
            if pos >= len(pred_tokens) or pred_tokens[pos] != gt_tokens[pos]:
                correct = False
                break

        if has_predicted_tokens:
            n_correct, n_total = results[cmd]
            results[cmd] = (n_correct + (1 if correct else 0), n_total + 1)

    return results
