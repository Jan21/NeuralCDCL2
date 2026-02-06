#!/usr/bin/env python3
"""
Generate CDCL solver training data.

This script generates traces from a CDCL SAT solver that can be used
to train an LLM to imitate solver behavior.

Usage:
    python generate.py              # Full dataset generation
    python generate.py --test       # Quick test with 100 formulas
"""

import argparse
import sys
from pathlib import Path

from cdcl_data_gen import (
    CDCLSolver,
    generate_random_formula,
    verify_solver_traces,
    collect_traces,
    create_datasets,
)


def main():
    parser = argparse.ArgumentParser(description="Generate CDCL solver training data")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Quick test mode with small dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory (default: output)"
    )
    parser.add_argument(
        "--iid-target",
        type=int,
        default=400000,
        help="Target traces for in-distribution dataset"
    )
    parser.add_argument(
        "--ood-target",
        type=int,
        default=10000,
        help="Target traces for out-of-distribution dataset"
    )
    parser.add_argument(
        "--test-size",
        type=int,
        default=2048,
        help="Number of test examples"
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip trace verification"
    )
    parser.add_argument(
        "--no-pysat",
        action="store_true",
        help="Skip PySAT verification"
    )

    args = parser.parse_args()

    if args.test:
        # Quick test mode
        print("=== TEST MODE ===")
        run_test(verify=not args.no_verify)
        return

    # Full dataset generation
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== GENERATING IN-DISTRIBUTION DATASET (5-15 variables) ===")
    iid_data = collect_traces(
        var_min=5,
        var_max=15,
        target_count=args.iid_target,
        verify=not args.no_verify,
        use_pysat_verification=not args.no_pysat
    )

    print(f"\nCollected {len(iid_data['solve_traces'])} solve traces")
    print(f"Collected {len(iid_data['up_traces'])} UP traces")
    print(f"Collected {len(iid_data['ac_traces'])} AC traces")
    print(f"Collected {len(iid_data['nested_traces'])} problems")

    create_datasets(iid_data, output_dir, test_size=args.test_size, prefix="iid_")

    print("\n=== GENERATING OUT-OF-DISTRIBUTION DATASET (16-25 variables) ===")
    ood_data = collect_traces(
        var_min=16,
        var_max=25,
        target_count=args.ood_target,
        verify=not args.no_verify,
        use_pysat_verification=not args.no_pysat
    )

    print(f"\nCollected {len(ood_data['solve_traces'])} solve traces")
    print(f"Collected {len(ood_data['up_traces'])} UP traces")
    print(f"Collected {len(ood_data['ac_traces'])} AC traces")
    print(f"Collected {len(ood_data['nested_traces'])} problems")

    create_datasets(ood_data, output_dir, test_size=args.test_size, prefix="ood_")

    print(f"\n=== GENERATION COMPLETE ===")
    print(f"Output directory: {output_dir}")


def run_test(verify: bool = True):
    """Run a quick test with a few formulas."""
    print("Testing solver with 100 random formulas...")

    success_count = 0
    verify_count = 0

    for i in range(100):
        # Generate formula
        n_vars = 5 + (i % 10)  # 5-14 variables
        clauses, variables = generate_random_formula(n_vars)

        # Solve
        solver = CDCLSolver(clauses, variables)
        result = solver.solve()

        # Verify trace reconstruction
        if verify:
            if verify_solver_traces(solver):
                verify_count += 1
            else:
                print(f"  Formula {i}: Verification FAILED")

        success_count += 1

    print(f"\nSolver: {success_count}/100 formulas solved")
    if verify:
        print(f"Verification: {verify_count}/100 traces valid")

    # Show example trace
    print("\n=== EXAMPLE TRACE ===")
    clauses, variables = generate_random_formula(5)
    solver = CDCLSolver(clauses, variables)
    solver.solve()

    if solver.nested_traces["solve_iterations"]:
        first_iter = solver.nested_traces["solve_iterations"][0]
        print("Solve trace (first iteration):")
        for line in first_iter["trace"][:10]:
            print(f"  {line}")
        if len(first_iter["trace"]) > 10:
            print(f"  ... ({len(first_iter['trace']) - 10} more lines)")

        if first_iter["subcalls"]:
            print("\nUnit propagation trace (first call):")
            up_trace = first_iter["subcalls"][0]["trace"]
            for line in up_trace[:10]:
                print(f"  {line}")
            if len(up_trace) > 10:
                print(f"  ... ({len(up_trace) - 10} more lines)")


if __name__ == "__main__":
    main()
