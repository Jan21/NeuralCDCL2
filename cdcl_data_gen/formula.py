"""
Random CNF formula generation for SAT solver training data.
"""

import random
from typing import List, Optional, Tuple


# Empirically determined clause counts for balanced (phase transition) 3-SAT
BALANCED_CLAUSE_COUNTS = {
    3: 19, 4: 24, 5: 28, 6: 33, 7: 37, 8: 41, 9: 45, 10: 50,
    11: 54, 12: 58, 13: 63, 14: 67, 15: 71, 16: 76, 17: 79,
    18: 83, 19: 87, 20: 92
}


def generate_random_formula(
    n_vars: int,
    clause_length: int = 3,
    variance: float = 0.1,
    n_clauses: Optional[int] = None
) -> Tuple[List[List[int]], range]:
    """
    Generate a random k-SAT formula near the phase transition.

    Args:
        n_vars: Number of variables.
        clause_length: Number of literals per clause (default 3 for 3-SAT).
        variance: Relative standard deviation in clause count (e.g., 0.1 = +/-10%).
        n_clauses: Fixed number of clauses. If None, uses phase transition estimate.

    Returns:
        Tuple of (clauses, variable_range) where:
        - clauses: List of clauses, each clause is a list of literals
        - variable_range: Range of variable indices used
    """
    if n_clauses is None:
        base = BALANCED_CLAUSE_COUNTS.get(n_vars, int(n_vars * 4.26))
        delta = int(base * variance)
        n_clauses = random.randint(base - delta, base + delta)

    # Random starting variable index (1-indexed, allows up to 25 vars total)
    interval_start = random.randint(1, 26 - n_vars)
    var_range = range(interval_start, interval_start + n_vars + 1)

    clauses = []
    for _ in range(n_clauses):
        clause_vars = random.sample(list(var_range), clause_length)
        clause = [var if random.random() < 0.5 else -var for var in clause_vars]
        clauses.append(clause)

    return clauses, var_range
