#!/usr/bin/env python3
"""
Test script for the CDCL environment with mock LLM.

Runs the environment with recorded traces to verify:
1. Environment correctly processes tokens from mock LLM
2. READ injections provide correct state data
3. WRITE commands update state correctly
4. Final state matches expected outcome
"""

import sys
from cdcl_data_gen import (
    CDCLSolver,
    generate_random_formula,
    CDCLEnvironment,
    MockLLM,
    TraceMismatchError,
)


def test_single_formula(verbose: bool = False):
    """Test environment with a single formula."""
    print("Testing single formula...")

    # Generate a formula
    clauses, variables = generate_random_formula(5)

    if verbose:
        print(f"  Clauses: {clauses}")
        print(f"  Variables: {list(variables)}")

    # Solve with original solver to get traces
    solver = CDCLSolver(clauses, variables)
    expected_result = solver.solve()

    if verbose:
        print(f"  Original solver result: {'SAT' if expected_result else 'UNSAT'}")
        print(f"  Iterations: {len(solver.nested_traces['solve_iterations'])}")

    # Now test the environment with mock LLM
    mock_llm = MockLLM(solver.nested_traces)
    env = CDCLEnvironment(clauses, mock_llm)

    try:
        env_result = env.solve()
        if verbose:
            print(f"  Environment result: {'SAT' if env_result else 'UNSAT'}")
            print(f"  Tokens processed: {env.token_count}")

        if env_result == expected_result:
            print("  PASS: Results match!")
            return True
        else:
            print(f"  FAIL: Results don't match! Expected {expected_result}, got {env_result}")
            return False

    except TraceMismatchError as e:
        print(f"  FAIL: Trace mismatch: {e}")
        return False
    except Exception as e:
        print(f"  FAIL: Unexpected error: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        return False


def test_multiple_formulas(count: int = 20, verbose: bool = False):
    """Test environment with multiple formulas."""
    print(f"\nTesting {count} formulas...")

    passed = 0
    failed = 0

    for i in range(count):
        n_vars = 5 + (i % 10)  # 5-14 variables
        clauses, variables = generate_random_formula(n_vars)

        # Solve with original solver
        solver = CDCLSolver(clauses, variables)
        expected_result = solver.solve()

        # Test with environment
        mock_llm = MockLLM(solver.nested_traces)
        env = CDCLEnvironment(clauses, mock_llm)

        try:
            env_result = env.solve()

            if env_result == expected_result:
                passed += 1
                if verbose:
                    print(f"  Formula {i+1}: PASS ({'SAT' if expected_result else 'UNSAT'})")
            else:
                failed += 1
                print(f"  Formula {i+1}: FAIL (expected {expected_result}, got {env_result})")

        except Exception as e:
            failed += 1
            print(f"  Formula {i+1}: ERROR - {e}")

    print(f"\nResults: {passed}/{count} passed, {failed}/{count} failed")
    return failed == 0


def test_state_reconstruction(verbose: bool = False):
    """Test that environment state matches solver state after execution."""
    print("\nTesting state reconstruction...")

    clauses, variables = generate_random_formula(8)

    # Solve with original solver
    solver = CDCLSolver(clauses, variables)
    solver.solve()

    # Get final state from solver
    final_iteration = solver.nested_traces["solve_iterations"][-1]
    expected_state = final_iteration["state_after"]

    if verbose:
        print(f"  Expected assignments: {expected_state['assignments']}")
        print(f"  Expected level: {expected_state['level']}")

    # Run environment
    mock_llm = MockLLM(solver.nested_traces)
    env = CDCLEnvironment(clauses, mock_llm)
    env.solve()

    # Compare states
    matches = True

    # Compare assignments
    if env.state.assignments != expected_state["assignments"]:
        print(f"  FAIL: Assignments mismatch")
        print(f"    Expected: {expected_state['assignments']}")
        print(f"    Got: {env.state.assignments}")
        matches = False

    # Compare level
    if env.state.level != expected_state["level"]:
        print(f"  FAIL: Level mismatch")
        print(f"    Expected: {expected_state['level']}")
        print(f"    Got: {env.state.level}")
        matches = False

    # Compare learned clauses (as sets)
    expected_learned = set(tuple(c) for c in expected_state["learned_clauses"])
    actual_learned = set(tuple(c) for c in env.state.learned_clauses)
    if expected_learned != actual_learned:
        print(f"  FAIL: Learned clauses mismatch")
        print(f"    Expected: {expected_learned}")
        print(f"    Got: {actual_learned}")
        matches = False

    if matches:
        print("  PASS: State matches!")
        return True
    return False


def test_parser():
    """Test the token parser."""
    print("\nTesting token parser...")
    from cdcl_data_gen import TokenParser, CommandType

    parser = TokenParser()

    # Test READ command
    cmd = parser.add_token("READ_ASSIGNMENTS")
    assert cmd is not None
    assert cmd.type == CommandType.READ
    assert cmd.name == "ASSIGNMENTS"
    print("  READ command parsing: PASS")

    # Test WRITE command
    parser.reset()
    assert parser.add_token("WRITE_ASSIGNMENTS") is None  # Waiting for [
    assert parser.add_token("[") is None  # Waiting for content
    assert parser.add_token("x") is None
    assert parser.add_token("1") is None
    assert parser.add_token("=") is None
    assert parser.add_token("True") is None
    cmd = parser.add_token("]")
    assert cmd is not None
    assert cmd.type == CommandType.WRITE
    assert cmd.name == "ASSIGNMENTS"
    assert "x 1 = True" in cmd.content
    print("  WRITE command parsing: PASS")

    # Test CALL command
    parser.reset()
    assert parser.add_token("CALL") is None  # Waiting for procedure
    cmd = parser.add_token("unit_propagate")
    assert cmd is not None
    assert cmd.type == CommandType.CALL
    assert cmd.name == "unit_propagate"
    print("  CALL command parsing: PASS")

    # Test END command
    parser.reset()
    cmd = parser.add_token("UNIT_PROPAGATION_END")
    assert cmd is not None
    assert cmd.type == CommandType.END
    print("  END command parsing: PASS")

    # Test BACKTRACK command
    parser.reset()
    assert parser.add_token("BACKTRACK") is None
    cmd = parser.add_token("0")
    assert cmd is not None
    assert cmd.type == CommandType.BACKTRACK
    assert cmd.name == "0"
    print("  BACKTRACK command parsing: PASS")

    print("  All parser tests: PASS")
    return True


def test_call_stack():
    """Test the call stack."""
    print("\nTesting call stack...")
    from cdcl_data_gen import CallStack

    stack = CallStack()

    assert stack.is_empty()
    assert stack.depth() == 0

    stack.push("solve")
    assert not stack.is_empty()
    assert stack.depth() == 1
    assert stack.current_procedure() == "solve"
    assert not stack.is_in_subcall()

    stack.push("unit_propagate")
    assert stack.depth() == 2
    assert stack.current_procedure() == "unit_propagate"
    assert stack.is_in_subcall()

    frame = stack.pop()
    assert frame.procedure == "unit_propagate"
    assert stack.depth() == 1
    assert not stack.is_in_subcall()

    print("  All call stack tests: PASS")
    return True


def main():
    """Run all tests."""
    print("=" * 50)
    print("CDCL Environment Tests")
    print("=" * 50)

    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    all_passed = True

    # Unit tests
    all_passed &= test_parser()
    all_passed &= test_call_stack()

    # Integration tests
    all_passed &= test_single_formula(verbose=verbose)
    all_passed &= test_multiple_formulas(count=20, verbose=verbose)
    all_passed &= test_state_reconstruction(verbose=verbose)

    print("\n" + "=" * 50)
    if all_passed:
        print("ALL TESTS PASSED")
    else:
        print("SOME TESTS FAILED")
    print("=" * 50)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
