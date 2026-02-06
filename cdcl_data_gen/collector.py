"""
Data collection and dataset creation for CDCL solver traces.
"""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from tqdm import tqdm

from .solver import CDCLSolver
from .formula import generate_random_formula
from .verifier import verify_solver_traces
from .format import fmt_int, fmt_clause


def collect_traces(
    var_min: int,
    var_max: int,
    target_count: int,
    verify: bool = True,
    batch_size: int = 10000,
    use_pysat_verification: bool = True
) -> Dict:
    """
    Collect solver traces until target count is reached.

    Args:
        var_min: Minimum number of variables.
        var_max: Maximum number of variables.
        target_count: Target number of each trace type.
        verify: Whether to verify trace reconstruction.
        batch_size: Number of formulas per batch.
        use_pysat_verification: Whether to verify against PySAT.

    Returns:
        Dictionary with traces and statistics.
    """
    all_solve_traces = []
    all_up_traces = []
    all_ac_traces = []
    all_nested_traces = []

    batch_num = 0
    verification_failures = 0
    pysat_available = False

    if use_pysat_verification:
        try:
            from pysat.solvers import Glucose3
            pysat_available = True
        except ImportError:
            print("Warning: PySAT not available, skipping external verification")

    while (len(all_solve_traces) < target_count or
           len(all_up_traces) < target_count or
           len(all_ac_traces) < target_count):

        print(f"Batch {batch_num + 1}: Generating {batch_size} formulas with {var_min}-{var_max} variables...")
        print(f"Current counts - Solve: {len(all_solve_traces)}, UP: {len(all_up_traces)}, AC: {len(all_ac_traces)}")

        # Generate formulas
        formulas = []
        for _ in range(batch_size):
            num_vars = random.randint(var_min, var_max)
            formulas.append(generate_random_formula(num_vars))

        # Process formulas
        for clauses, variables in tqdm(formulas, desc="Processing formulas"):
            # Verify with PySAT if available
            if pysat_available:
                from pysat.solvers import Glucose3
                g = Glucose3()
                for clause in clauses:
                    g.add_clause(clause)
                is_sat_pysat = g.solve()
                g.delete()

            # Run our solver
            solver = CDCLSolver(clauses, variables)
            is_satisfiable = solver.solve()

            # Verify result matches PySAT
            if pysat_available:
                assert is_sat_pysat == is_satisfiable, "Result mismatch with PySAT"

                if is_satisfiable:
                    # Verify solution
                    g_verify = Glucose3()
                    for clause in clauses:
                        g_verify.add_clause(clause)
                    assumptions = [var if val else -var for var, val in solver.assignments.items()]
                    assert g_verify.solve(assumptions=assumptions), "Invalid solution"
                    g_verify.delete()

            # Verify trace reconstruction
            if verify:
                if not verify_solver_traces(solver):
                    verification_failures += 1
                    print(f"Warning: Trace verification failed (total: {verification_failures})")

            # Store nested traces
            solver.nested_traces["input_clauses"] = {
                str(k): v for k, v in solver.clause2id.items()
            }
            all_nested_traces.append(solver.nested_traces)

            # Extract individual traces
            for iteration in solver.nested_traces["solve_iterations"]:
                if len(all_solve_traces) < target_count:
                    all_solve_traces.append(_format_trace(iteration["trace"]))

                for subcall in iteration.get("subcalls", []):
                    if subcall["procedure"] == "unit_propagate":
                        if len(all_up_traces) < target_count:
                            all_up_traces.append(_format_trace(subcall["trace"]))
                    elif subcall["procedure"] == "analyze_conflict":
                        if len(all_ac_traces) < target_count:
                            all_ac_traces.append(_format_trace(subcall["trace"]))

        batch_num += 1

    if verify:
        print(f"Verification complete. Total failures: {verification_failures}")

    return {
        "solve_traces": all_solve_traces[:target_count],
        "up_traces": all_up_traces[:target_count],
        "ac_traces": all_ac_traces[:target_count],
        "nested_traces": all_nested_traces,
        "verification_failures": verification_failures,
    }


def _format_trace(trace: List) -> str:
    """Convert trace list to string."""
    result = []
    for item in trace:
        if isinstance(item, list):
            result.extend(str(sub) for sub in item)
        else:
            result.append(str(item))
    return ' '.join(result)


def create_datasets(
    data: Dict,
    output_dir: str,
    test_size: int = 2048,
    prefix: str = ""
) -> None:
    """
    Save collected traces to dataset files.

    Args:
        data: Dictionary from collect_traces().
        output_dir: Output directory path.
        test_size: Number of test examples.
        prefix: Prefix for filenames (e.g., "iid_" or "ood_").
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    solve_traces = data["solve_traces"]
    up_traces = data["up_traces"]
    ac_traces = data["ac_traces"]
    nested_traces = data["nested_traces"]

    # Create mixed dataset
    mixed_data = []
    for trace in solve_traces:
        mixed_data.append({"text": trace})
    for trace in up_traces:
        mixed_data.append({"text": trace})
    for trace in ac_traces:
        mixed_data.append({"text": trace})
    random.shuffle(mixed_data)

    # Split train/test
    test_data = mixed_data[:test_size]
    train_data = mixed_data[test_size:]

    # Save files
    with open(output_path / f"{prefix}train.json", 'w') as f:
        json.dump(train_data, f, indent=2)

    with open(output_path / f"{prefix}test.json", 'w') as f:
        json.dump(test_data, f, indent=2)

    with open(output_path / f"{prefix}test_solve.json", 'w') as f:
        json.dump([{"text": t} for t in solve_traces[:test_size]], f, indent=2)

    with open(output_path / f"{prefix}test_up.json", 'w') as f:
        json.dump([{"text": t} for t in up_traces[:test_size]], f, indent=2)

    with open(output_path / f"{prefix}test_ac.json", 'w') as f:
        json.dump([{"text": t} for t in ac_traces[:test_size]], f, indent=2)

    with open(output_path / f"{prefix}nested_traces.json", 'w') as f:
        json.dump(nested_traces, f, indent=2)

    print(f"Saved datasets to {output_path}")
    print(f"  Train: {len(train_data)} examples")
    print(f"  Test: {len(test_data)} examples")
    print(f"  Nested traces: {len(nested_traces)} problems")
