"""
CDCL SAT Solver with trace generation for LLM training data.

The solver generates traces that can be used to train an LLM to imitate
the behavior of a CDCL solver. The trace format uses:
- READ_X [ content ] for reading state
- WRITE_X [ content ] for writing state
- CALL procedure_name for calling subprocedures
- BACKTRACK for backtracking
"""

from typing import Dict, List, Optional, Tuple

from .format import (
    fmt_int, fmt_var, fmt_lit, fmt_clause, fmt_clause_list,
    fmt_assignments, fmt_decision_levels, fmt_reason_clauses,
    fmt_lit_list, fmt_var_list, fmt_level_list
)


class CDCLSolver:
    """
    CDCL SAT Solver that generates execution traces.

    The solver maintains:
    - assignments: variable -> bool mapping
    - decision_level: variable -> level at which it was assigned
    - reason_clauses: variable -> clause that implied its value
    - learned_clauses: clauses learned through conflict analysis
    """

    def __init__(self, clauses: List[List[int]], variables: range):
        # Input
        self.clauses = clauses
        self.variables = variables

        # State
        self.assignments: Dict[int, bool] = {}
        self.decision_level: Dict[int, int] = {}
        self.reason_clauses: Dict[int, List[int]] = {}
        self.learned_clauses: List[List[int]] = []
        self.level: int = 0

        # Clause ID mapping
        self.clause2id = {tuple(c): f"c {fmt_int(i)}" for i, c in enumerate(clauses)}

        # Trace storage
        self.nested_traces = {
            "input_clauses": {},
            "solve_iterations": []
        }
        self._current_iteration: Optional[dict] = None

    def solve(self) -> bool:
        """
        Main CDCL solve loop.

        Returns True if satisfiable, False if unsatisfiable.
        """
        iteration_index = 0

        while True:
            # Create nested trace for this iteration
            solve_iter = {
                "procedure": "solve",
                "iteration": iteration_index,
                "trace": [],
                "subcalls": [],
                "state_before": self._capture_state(),
                "outcome": None,
            }
            self._current_iteration = solve_iter
            trace = []

            # Begin solve iteration
            trace.append("SOLVE_BEGIN")
            trace.append("CALL unit_propagate")

            # Unit propagation
            conflict = self._unit_propagate()

            # Read current state
            trace.append(f"READ_ASSIGNMENTS [ {fmt_assignments(self.assignments)} ]")
            trace.append(f"READ_CLAUSES [ {fmt_clause_list(self.clauses + self.learned_clauses, self.clause2id)} ]")
            trace.append(f"READ_DECISION_LEVELS [ {fmt_decision_levels(self.decision_level)} ]")
            trace.append(f"READ_LEVEL [ {fmt_int(self.level)} ]")

            if conflict:
                # Conflict found
                trace.append(f"READ_CONFLICT_CLAUSE [ {fmt_clause(conflict, self.clause2id[tuple(conflict)])} ]")

                if self.level == 0:
                    # Conflict at level 0 means UNSAT
                    trace.append("UNSAT")
                    trace.append("SOLVE_END")
                    solve_iter["trace"] = trace
                    solve_iter["outcome"] = "UNSAT"
                    solve_iter["state_after"] = self._capture_state()
                    self.nested_traces["solve_iterations"].append(solve_iter)
                    return False

                # Analyze conflict
                trace.append("CALL analyze_conflict")
                learned_clause = self._analyze_conflict(conflict)
                trace.append(f"READ_LEARNED_CLAUSE [ {fmt_clause(learned_clause)} ]")

                # Find backtrack level
                backtrack_level, bt_trace = self._find_backtrack_level(learned_clause)
                trace.extend(bt_trace)

                # Backtrack
                trace.append(f"BACKTRACK {fmt_int(backtrack_level)}")
                self._backtrack(backtrack_level)

                # Add learned clause
                self.learned_clauses.append(learned_clause)
                self.clause2id[tuple(learned_clause)] = f"c {fmt_int(len(self.clause2id))}"

            else:
                # No conflict
                trace.append("NO_CONFLICT")
                trace.append(f"N_ASSIGNED [ {fmt_int(len(self.assignments))} ]")

                if len(self.assignments) == self._count_variables():
                    # All variables assigned - SAT
                    trace.append("SAT")
                    trace.append("SOLVE_END")
                    solve_iter["trace"] = trace
                    solve_iter["outcome"] = "SAT"
                    solve_iter["state_after"] = self._capture_state()
                    self.nested_traces["solve_iterations"].append(solve_iter)
                    return True

                # Pick branching variable
                var = self._pick_branching_variable()
                trace.append(f"BRANCHING_VARIABLE [ {fmt_var(var)} ]")

                if var is None:
                    trace.append("SAT")
                    trace.append("SOLVE_END")
                    solve_iter["trace"] = trace
                    solve_iter["outcome"] = "SAT"
                    solve_iter["state_after"] = self._capture_state()
                    self.nested_traces["solve_iterations"].append(solve_iter)
                    return True

                # Make decision
                self.level += 1
                self._assign(var, True, None)
                trace.append(f"WRITE_ASSIGNMENTS [ {fmt_var(var)} = True ]")
                trace.append("LEVEL_UP")

            trace.append("SOLVE_ITER_END")
            solve_iter["trace"] = trace
            solve_iter["outcome"] = "CONTINUE"
            solve_iter["state_after"] = self._capture_state()
            self.nested_traces["solve_iterations"].append(solve_iter)
            iteration_index += 1

    def _unit_propagate(self) -> Optional[List[int]]:
        """
        Unit propagation.

        Returns the conflict clause if a conflict is found, None otherwise.
        """
        trace = []
        trace.append("UNIT_PROPAGATION_BEGIN")
        trace.append(f"READ_ASSIGNMENTS [ {fmt_assignments(self.assignments)} ]")
        trace.append(f"READ_CLAUSES [ {fmt_clause_list(self.clauses + self.learned_clauses, self.clause2id)} ]")

        while True:
            propagated = False

            for clause in self.clauses + self.learned_clauses:
                all_assigned, satisfied = self._evaluate_clause(clause)
                trace.append(f"EVALUATE_CLAUSE {fmt_clause(clause, self.clause2id[tuple(clause)])}")

                if all_assigned and not satisfied:
                    # Conflict: all literals assigned but clause unsatisfied
                    trace.append(f"WRITE_CONFLICT_CLAUSE [ {self.clause2id[tuple(clause)]} ]")
                    trace.append("UNIT_PROPAGATION_END")

                    if self._current_iteration:
                        self._current_iteration["subcalls"].append({
                            "procedure": "unit_propagate",
                            "trace": trace,
                            "subcalls": [],
                            "result": {"conflict_clause": list(clause)}
                        })
                    return clause

                elif self._is_unit(clause):
                    # Unit clause: propagate
                    trace.append("PROPAGATE")
                    lit = self._get_unassigned_literal(clause)
                    var = abs(lit)
                    value = lit > 0
                    propagated = True

                    trace.append(f"WRITE_ASSIGNMENTS [ {fmt_var(var)} = {value} BECAUSE {self.clause2id[tuple(clause)]} ]")
                    self._assign(var, value, clause)

            if not propagated:
                trace.append("NO_PROPAGATION")
                break

        trace.append("UNIT_PROPAGATION_END")

        if self._current_iteration:
            self._current_iteration["subcalls"].append({
                "procedure": "unit_propagate",
                "trace": trace,
                "subcalls": [],
                "result": {"conflict_clause": None}
            })
        return None

    def _analyze_conflict(self, conflict_clause: List[int]) -> List[int]:
        """
        Conflict analysis using first UIP scheme.

        Returns the learned clause.
        """
        trace = []
        trace.append("ANALYZE_CONFLICT_BEGIN")
        trace.append(f"READ_ASSIGNMENTS [ {fmt_assignments(self.assignments)} ]")
        trace.append(f"READ_CLAUSES [ {fmt_clause_list(self.clauses + self.learned_clauses, self.clause2id)} ]")
        trace.append(f"READ_DECISION_LEVELS [ {fmt_decision_levels(self.decision_level)} ]")
        trace.append(f"READ_LEVEL [ {fmt_int(self.level)} ]")
        trace.append(f"READ_REASON_CLAUSES [ {fmt_reason_clauses(self.reason_clauses, self.clause2id)} ]")
        trace.append(f"READ_CONFLICT_CLAUSE [ {fmt_clause(conflict_clause, self.clause2id[tuple(conflict_clause)])} ]")

        # Initialize tracking sets
        current_level_vars = set()  # Variables assigned at current decision level
        learned_lits = set()        # Literals for learned clause (from earlier levels)

        # Start with conflict clause literals
        queue = list(conflict_clause)
        trace.append(f"QUEUE [ {fmt_lit_list(queue)} ]")

        while True:
            # Process each literal in queue
            for lit in queue:
                var = abs(lit)
                var_level = self.decision_level.get(var)
                trace.append(f"CHECK_VAR [ {fmt_var(var)} AT_LEVEL {fmt_int(var_level)} ]")

                if var_level == self.level:
                    # Variable at current level
                    current_level_vars.add(var)
                    trace.append(f"CURRENT_LEVEL_VARS [ {fmt_var_list(list(current_level_vars))} ]")
                else:
                    # Variable at earlier level - add to learned clause
                    learned_lit = -var if self.assignments[var] else var
                    learned_lits.add(learned_lit)
                    trace.append(f"LEARNED_LIT [ {fmt_lit(learned_lit)} ]")

            # UIP condition: only one variable from current level remains
            if len(current_level_vars) <= 1:
                trace.append("UIP")
                break

            # Get most recently assigned variable from current level
            var = self._get_latest_assigned(current_level_vars)
            trace.append(f"LATEST_ASSIGNED [ {fmt_var(var)} ]")
            current_level_vars.remove(var)

            # Get reason clause and resolve
            reason = self.reason_clauses.get(var)
            trace.append(f"REASON [ {fmt_var(var)} -> {self.clause2id[tuple(reason)]} ]")

            if reason:
                queue = [lit for lit in reason if abs(lit) != var]
                trace.append(f"QUEUE [ {fmt_lit_list(queue)} ]")

        # Create learned clause
        current_level_lits = [-var if self.assignments[var] else var for var in current_level_vars]
        trace.append(f"CURRENT_LEVEL_LITS [ {fmt_lit_list(current_level_lits)} ]")

        new_clause = list(learned_lits.union(set(current_level_lits)))
        trace.append(f"WRITE_LEARNED_CLAUSE [ {fmt_clause(new_clause)} ]")
        trace.append("ANALYZE_CONFLICT_END")

        if self._current_iteration:
            self._current_iteration["subcalls"].append({
                "procedure": "analyze_conflict",
                "trace": trace,
                "subcalls": [],
                "result": {"learned_clause": list(new_clause)}
            })

        return new_clause

    def _find_backtrack_level(self, learned_clause: List[int]) -> Tuple[int, List[str]]:
        """
        Find the backtrack level for a learned clause.

        Returns (backtrack_level, trace).
        """
        trace = []
        trace.append(f"READ_DECISION_LEVELS [ {fmt_decision_levels(self.decision_level)} ]")

        levels = [self.decision_level[abs(lit)] for lit in learned_clause
                  if abs(lit) in self.decision_level]
        trace.append(f"LEVELS {fmt_level_list(levels)}")

        if not levels:
            #trace.append(f"WRITE_BACKTRACK_LEVEL [ {fmt_int(0)} ]")
            return 0, trace

        levels.sort(reverse=True)
        goto = levels[1] if len(levels) > 1 else 0
        #trace.append(f"WRITE_BACKTRACK_LEVEL [ {fmt_int(goto)} ]")

        return goto, trace

    def _backtrack(self, level: int):
        """Backtrack to the given decision level."""
        self.level = level
        self.assignments = {
            var: val for var, val in self.assignments.items()
            if self.decision_level[var] <= level
        }
        self.reason_clauses = {
            var: clause for var, clause in self.reason_clauses.items()
            if var in self.assignments
        }

    def _assign(self, var: int, value: bool, reason: Optional[List[int]]):
        """Assign a value to a variable."""
        self.assignments[var] = value
        self.decision_level[var] = self.level
        if reason:
            self.reason_clauses[var] = reason

    def _evaluate_clause(self, clause: List[int]) -> Tuple[bool, bool]:
        """
        Evaluate a clause under current assignments.

        Returns (all_assigned, satisfied).
        """
        satisfied = False
        all_assigned = True

        for lit in clause:
            var = abs(lit)
            if var in self.assignments:
                if (lit > 0) == self.assignments[var]:
                    satisfied = True
                    break
            else:
                all_assigned = False

        return all_assigned, satisfied

    def _is_unit(self, clause: List[int]) -> bool:
        """Check if clause is unit (one unassigned literal, not satisfied)."""
        unassigned = 0
        for lit in clause:
            var = abs(lit)
            if var in self.assignments:
                if (lit > 0) == self.assignments[var]:
                    return False  # Satisfied
            else:
                unassigned += 1
        return unassigned == 1

    def _get_unassigned_literal(self, clause: List[int]) -> Optional[int]:
        """Get the unassigned literal in a unit clause."""
        for lit in clause:
            if abs(lit) not in self.assignments:
                return lit
        return None

    def _get_latest_assigned(self, vars_set: set) -> int:
        """Get the most recently assigned variable from a set."""
        return max(vars_set, key=lambda var: list(self.assignments.keys()).index(var))

    def _pick_branching_variable(self) -> Optional[int]:
        """Pick the next unassigned variable for branching."""
        for var in self.variables:
            if var not in self.assignments:
                return var
        return None

    def _count_variables(self) -> int:
        """Count unique variables in the formula."""
        vars_set = set()
        for clause in self.clauses:
            for lit in clause:
                vars_set.add(abs(lit))
        return len(vars_set)

    def _capture_state(self) -> dict:
        """Capture current solver state for verification."""
        return {
            "assignments": dict(self.assignments),
            "decision_level": dict(self.decision_level),
            "reason_clauses": {k: list(v) for k, v in self.reason_clauses.items()},
            "level": self.level,
            "learned_clauses": [list(c) for c in self.learned_clauses],
        }
