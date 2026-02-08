"""
Loss mask computation for CDCL traces.

Masking rules:
1. Mask the first token (position 0) — the BEGIN command
2. Mask tokens inside [...] brackets that follow READ_* commands (inclusive of [ and ])
3. Everything else is NOT masked (loss is computed)
"""

from typing import List

# READ commands that trigger environment data injection.
# Duplicated here from cdcl_data_gen.parser to avoid heavy __init__ imports.
READ_COMMANDS = {
    "READ_ASSIGNMENTS",
    "READ_CLAUSES",
    "READ_DECISION_LEVELS",
    "READ_LEVEL",
    "READ_CONFLICT_CLAUSE",
    "READ_REASON_CLAUSES",
    "READ_LEARNED_CLAUSE",
}


def compute_loss_mask(tokens: List[str]) -> List[int]:
    """
    Compute loss mask for a token sequence.

    Args:
        tokens: List of whitespace-split tokens from a trace.

    Returns:
        List of 0/1 values, same length as tokens.
        0 = do NOT compute loss (masked), 1 = compute loss.
    """
    n = len(tokens)
    mask = [1] * n

    if n == 0:
        return mask

    # Rule 1: mask the first token
    mask[0] = 0

    # Rule 2: mask [...] content after READ_* commands
    i = 0
    while i < n:
        if tokens[i] in READ_COMMANDS:
            # The READ token itself stays unmasked (mask=1)
            # Look for the following [ ... ]
            j = i + 1
            if j < n and tokens[j] == "[":
                depth = 0
                while j < n:
                    if tokens[j] == "[":
                        depth += 1
                    elif tokens[j] == "]":
                        depth -= 1
                    mask[j] = 0
                    if depth == 0:
                        break
                    j += 1
                i = j + 1
            else:
                i += 1
        else:
            i += 1

    return mask
