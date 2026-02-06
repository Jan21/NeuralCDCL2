"""
State reconstruction verification for CDCL solver traces.

Verifies that traces contain sufficient information to reconstruct
the solver state at each step.
"""

import re
from typing import Dict, List, Optional


class TraceVerifier:
    """
    Verifies that nested traces allow complete state reconstruction.

    The verifier replays WRITE operations from traces and compares
    the reconstructed state with the actual state captured by the solver.
    """

    def __init__(self, nested_traces: dict):
        self.traces = nested_traces

    def verify(self) -> bool:
        """
        Verify all solve iterations.

        Returns True if all traces are consistent, False otherwise.
        """
        for solve_iter in self.traces.get("solve_iterations", []):
            if not self._verify_iteration(solve_iter):
                return False
        return True

    def _verify_iteration(self, solve_iter: dict) -> bool:
        """Verify a single solve iteration."""
        state_before = solve_iter["state_before"]
        state_after = solve_iter["state_after"]

        # Start with state_before
        reconstructed = {
            "assignments": dict(state_before["assignments"]),
            "decision_level": dict(state_before["decision_level"]),
            "reason_clauses": {k: list(v) for k, v in state_before["reason_clauses"].items()},
            "level": state_before["level"],
            "learned_clauses": [list(c) for c in state_before["learned_clauses"]],
        }

        # Replay subcalls
        for subcall in solve_iter.get("subcalls", []):
            procedure = subcall["procedure"]
            trace_str = self._join_trace(subcall["trace"])

            if procedure == "unit_propagate":
                reconstructed = self._replay_unit_propagate(trace_str, reconstructed)
            elif procedure == "analyze_conflict":
                reconstructed = self._replay_analyze_conflict(trace_str, reconstructed)

        # Replay solve-level operations
        solve_trace_str = self._join_trace(solve_iter["trace"])
        reconstructed = self._replay_solve_trace(solve_trace_str, reconstructed)

        # Compare
        return self._states_equal(reconstructed, state_after)

    def _join_trace(self, trace: List) -> str:
        """Join trace list into string."""
        result = []
        for item in trace:
            if isinstance(item, list):
                result.extend(str(sub) for sub in item)
            else:
                result.append(str(item))
        return ' '.join(result)

    def _parse_spaced_int(self, text: str) -> int:
        """Parse space-separated digits back to int (e.g., '1 2' -> 12)."""
        return int(text.replace(' ', '').strip())

    def _replay_unit_propagate(self, trace_str: str, state: dict) -> dict:
        """Replay unit propagation trace to update state."""
        # Find all WRITE_ASSIGNMENTS operations with new bracket format
        pattern = r'WRITE_ASSIGNMENTS\s*\[\s*(.*?)\s*\]'
        for match in re.finditer(pattern, trace_str):
            content = match.group(1)
            # Parse: x <var> = <value> BECAUSE <clause_id>
            assign_pattern = r'x\s+([\d\s]+)\s*=\s*(True|False)(?:\s+BECAUSE\s+(c\s*[\d\s]+))?'
            assign_match = re.search(assign_pattern, content)
            if assign_match:
                var = self._parse_spaced_int(assign_match.group(1))
                value = assign_match.group(2) == 'True'
                state["assignments"][var] = value
                state["decision_level"][var] = state["level"]
        return state

    def _replay_analyze_conflict(self, trace_str: str, state: dict) -> dict:
        """Replay analyze conflict trace to update state."""
        # Find WRITE_LEARNED_CLAUSE with new bracket format
        pattern = r'WRITE_LEARNED_CLAUSE\s*\[\s*(.*?)\s*\]'
        match = re.search(pattern, trace_str)
        if match:
            content = match.group(1)
            clause = self._parse_clause(content)
            if clause:
                state["learned_clauses"].append(clause)
        return state

    def _parse_clause(self, content: str) -> List[int]:
        """Parse clause string like '( + x 1 - x 2 )' to list of literals."""
        literals = []
        pattern = r'([+-])\s*x\s*([\d\s]+)'
        for match in re.finditer(pattern, content):
            sign = 1 if match.group(1) == '+' else -1
            var = self._parse_spaced_int(match.group(2))
            literals.append(sign * var)
        return literals

    def _replay_solve_trace(self, trace_str: str, state: dict) -> dict:
        """Replay solve trace operations (backtrack, level changes, branching)."""
        # Check for BACKTRACK with level
        backtrack_pattern = r'BACKTRACK\s+([\d\s]+)'
        bt_match = re.search(backtrack_pattern, trace_str)
        if bt_match:
            bt_level = self._parse_spaced_int(bt_match.group(1))
            state["level"] = bt_level
            # Remove assignments above backtrack level
            state["assignments"] = {
                var: val for var, val in state["assignments"].items()
                if state["decision_level"].get(var, 0) <= bt_level
            }
            state["decision_level"] = {
                var: lvl for var, lvl in state["decision_level"].items()
                if var in state["assignments"]
            }
            state["reason_clauses"] = {
                var: clause for var, clause in state["reason_clauses"].items()
                if var in state["assignments"]
            }

        # Check for LEVEL_UP (branching)
        if "LEVEL_UP" in trace_str:
            state["level"] += 1
            # Find branching assignment
            assign_pattern = r'WRITE_ASSIGNMENTS\s*\[\s*x\s*([\d\s]+)\s*=\s*(True|False)\s*\]'
            assign_match = re.search(assign_pattern, trace_str)
            if assign_match:
                var = self._parse_spaced_int(assign_match.group(1))
                value = assign_match.group(2) == 'True'
                state["assignments"][var] = value
                state["decision_level"][var] = state["level"]

        return state

    def _states_equal(self, state1: dict, state2: dict) -> bool:
        """Compare two state dictionaries for equality."""
        if state1["assignments"] != state2["assignments"]:
            return False

        # Only compare decision_level for currently assigned variables
        dl1 = {k: v for k, v in state1["decision_level"].items() if k in state1["assignments"]}
        dl2 = {k: v for k, v in state2["decision_level"].items() if k in state2["assignments"]}
        if dl1 != dl2:
            return False

        if state1["level"] != state2["level"]:
            return False

        # Compare learned clauses as sets of tuples
        lc1 = set(tuple(c) for c in state1["learned_clauses"])
        lc2 = set(tuple(c) for c in state2["learned_clauses"])
        if lc1 != lc2:
            return False

        return True


def verify_solver_traces(solver) -> bool:
    """
    Verify traces from a CDCLSolver instance.

    Args:
        solver: CDCLSolver instance after solve() has been called.

    Returns:
        True if all traces are valid, False otherwise.
    """
    verifier = TraceVerifier(solver.nested_traces)
    return verifier.verify()
