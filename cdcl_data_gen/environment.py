"""
CDCL Environment that communicates with an LLM.

The environment maintains solver state and processes commands from the LLM,
injecting READ data and tracking WRITE operations.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .call_stack import CallStack
from .errors import InvalidCommandError
from .format import (
    fmt_int, fmt_var, fmt_clause, fmt_clause_list,
    fmt_assignments, fmt_decision_levels, fmt_reason_clauses
)
from .llm_interface import LLMInterface
from .parser import (
    TokenParser, Command, CommandType,
    parse_write_assignments, parse_write_learned_clause
)


@dataclass
class SolverState:
    """
    Maintains the current state of the CDCL solver.

    This state is manipulated by WRITE commands from the LLM
    and read by READ commands.
    """
    clauses: List[List[int]] = field(default_factory=list)
    assignments: Dict[int, bool] = field(default_factory=dict)
    decision_level: Dict[int, int] = field(default_factory=dict)
    reason_clauses: Dict[int, List[int]] = field(default_factory=dict)
    learned_clauses: List[List[int]] = field(default_factory=list)
    level: int = 0
    clause2id: Dict[Tuple[int, ...], str] = field(default_factory=dict)
    conflict_clause: Optional[List[int]] = None
    last_learned_clause: Optional[List[int]] = None
    result: Optional[bool] = None  # SAT result when determined

    def all_clauses(self) -> List[List[int]]:
        """Return all clauses (input + learned)."""
        return self.clauses + self.learned_clauses

    def backtrack(self, target_level: int) -> None:
        """
        Backtrack to the given decision level.

        Removes all assignments made at levels higher than target_level.
        """
        self.level = target_level
        # Remove assignments above target level
        vars_to_remove = [
            var for var, lvl in self.decision_level.items()
            if lvl > target_level
        ]
        for var in vars_to_remove:
            del self.assignments[var]
            del self.decision_level[var]
            if var in self.reason_clauses:
                del self.reason_clauses[var]

    def assign(self, var: int, value: bool, reason: Optional[str] = None) -> None:
        """
        Assign a value to a variable at the current level.

        Args:
            var: Variable to assign.
            value: Value to assign (True/False).
            reason: Clause ID that implied this assignment (for propagations).
        """
        self.assignments[var] = value
        self.decision_level[var] = self.level

        if reason:
            # Look up the clause from its ID
            for clause, cid in self.clause2id.items():
                if cid == reason:
                    self.reason_clauses[var] = list(clause)
                    break

    def add_learned_clause(self, clause: List[int]) -> None:
        """Add a learned clause to the solver state."""
        self.learned_clauses.append(clause)
        self.last_learned_clause = clause
        # Assign clause ID
        new_id = f"c {fmt_int(len(self.clause2id))}"
        self.clause2id[tuple(clause)] = new_id

    def copy(self) -> "SolverState":
        """Create a deep copy of the state."""
        return SolverState(
            clauses=[list(c) for c in self.clauses],
            assignments=dict(self.assignments),
            decision_level=dict(self.decision_level),
            reason_clauses={k: list(v) for k, v in self.reason_clauses.items()},
            learned_clauses=[list(c) for c in self.learned_clauses],
            level=self.level,
            clause2id=dict(self.clause2id),
            conflict_clause=list(self.conflict_clause) if self.conflict_clause else None,
            last_learned_clause=list(self.last_learned_clause) if self.last_learned_clause else None,
            result=self.result
        )


class CDCLEnvironment:
    """
    Environment that communicates with an LLM to simulate CDCL solver behavior.

    The environment:
    1. Processes tokens from the LLM
    2. Detects commands (READ, WRITE, CALL, BACKTRACK, etc.)
    3. Maintains solver state
    4. Injects data for READ commands
    5. Manages nested procedure calls
    """

    def __init__(self, clauses: List[List[int]], llm: LLMInterface):
        """
        Initialize the environment.

        Args:
            clauses: Input CNF clauses.
            llm: LLM interface for generation and injection.
        """
        self.llm = llm
        self.state = SolverState(clauses=[list(c) for c in clauses])

        # Initialize clause IDs
        for i, clause in enumerate(clauses):
            self.state.clause2id[tuple(clause)] = f"c {fmt_int(i)}"

        self.call_stack = CallStack()
        self.parser = TokenParser()

        # Tracking
        self.token_count = 0
        self.completed = False

    def solve(self) -> bool:
        """
        Run the solver using the LLM for decisions.

        Returns:
            True if SAT, False if UNSAT.
        """
        # Start solve procedure
        self.call_stack.push("solve", iteration=0)
        self.llm.inject("SOLVE_BEGIN")

        # Main loop - process tokens from LLM
        for token in self.llm.generate(""):
            self.token_count += 1
            command = self.parser.add_token(token)

            if command:
                self._handle_command(command)

            if self.completed:
                break

        return self.state.result if self.state.result is not None else False

    def _handle_command(self, command: Command) -> None:
        """
        Handle a parsed command from the LLM.

        Args:
            command: The parsed Command object.
        """
        if command.type == CommandType.READ:
            self._handle_read(command)
        elif command.type == CommandType.WRITE:
            self._handle_write(command)
        elif command.type == CommandType.CALL:
            self._handle_call(command)
        elif command.type == CommandType.END:
            self._handle_end(command)
        elif command.type == CommandType.BACKTRACK:
            self._handle_backtrack(command)
        elif command.type == CommandType.OTHER:
            # Handle special tokens
            if command.name == "SAT":
                self.state.result = True
            elif command.name == "UNSAT":
                self.state.result = False
            elif command.name == "LEVEL_UP":
                self.state.level += 1

    def _handle_read(self, command: Command) -> None:
        """
        Handle READ command - inject data from state.

        Args:
            command: The READ command.
        """
        content = self._get_read_content(command.name)
        self.llm.inject(f"[ {content} ]")

    def _get_read_content(self, name: str) -> str:
        """
        Get content for a READ command based on current state.

        Args:
            name: The read type (ASSIGNMENTS, CLAUSES, etc.)

        Returns:
            Formatted content string.
        """
        if name == "ASSIGNMENTS":
            return fmt_assignments(self.state.assignments)
        elif name == "CLAUSES":
            return fmt_clause_list(self.state.all_clauses(), self.state.clause2id)
        elif name == "DECISION_LEVELS":
            return fmt_decision_levels(self.state.decision_level)
        elif name == "LEVEL":
            return fmt_int(self.state.level)
        elif name == "CONFLICT_CLAUSE":
            if self.state.conflict_clause:
                cid = self.state.clause2id[tuple(self.state.conflict_clause)]
                return fmt_clause(self.state.conflict_clause, cid)
            return ""
        elif name == "REASON_CLAUSES":
            return fmt_reason_clauses(self.state.reason_clauses, self.state.clause2id)
        elif name == "LEARNED_CLAUSE":
            if self.state.last_learned_clause:
                return fmt_clause(self.state.last_learned_clause)
            return ""
        else:
            raise InvalidCommandError(f"READ_{name}", f"Unknown read type: {name}")

    def _handle_write(self, command: Command) -> None:
        """
        Handle WRITE command - update state.

        Args:
            command: The WRITE command with content.
        """
        if command.name == "ASSIGNMENTS":
            parsed = parse_write_assignments(command.content)
            if parsed:
                self.state.assign(
                    parsed["var"],
                    parsed["value"],
                    parsed.get("reason")
                )
        elif command.name == "CONFLICT_CLAUSE":
            # Look up clause from ID
            clause_id = command.content.strip()
            for clause, cid in self.state.clause2id.items():
                if cid == clause_id:
                    self.state.conflict_clause = list(clause)
                    break
        elif command.name == "LEARNED_CLAUSE":
            clause = parse_write_learned_clause(command.content)
            if clause:
                self.state.add_learned_clause(clause)
        else:
            raise InvalidCommandError(f"WRITE_{command.name}", f"Unknown write type")

    def _handle_call(self, command: Command) -> None:
        """
        Handle CALL command - start subcall.

        Args:
            command: The CALL command.
        """
        procedure = command.name

        # Push new frame
        self.call_stack.push(procedure)

        # Notify LLM of subcall
        self.llm.start_subcall(procedure)

        # Inject BEGIN token
        if procedure == "unit_propagate":
            self.llm.inject("UNIT_PROPAGATION_BEGIN")
        elif procedure == "analyze_conflict":
            self.llm.inject("ANALYZE_CONFLICT_BEGIN")

        # Reset parser for new procedure
        self.parser.reset()

    def _handle_end(self, command: Command) -> None:
        """
        Handle END command - end current procedure.

        Args:
            command: The END command.
        """
        end_type = command.name

        if end_type == "solve_iter":
            # End of solve iteration - increment iteration and continue
            current_iter = self.call_stack.get_iteration()
            self.call_stack.set_iteration(current_iter + 1)
            # Reset subcall index for new iteration
            self.call_stack.peek().subcall_index = 0
            # Inject SOLVE_BEGIN for next iteration
            self.llm.inject("SOLVE_BEGIN")
            self.parser.reset()

        elif end_type == "solve":
            # End of solve - we're done
            self.completed = True

        elif end_type in ("unit_propagation", "analyze_conflict"):
            # End of subcall - pop stack and return to parent
            self.call_stack.pop()
            self.llm.end_subcall()

            # Increment subcall index in parent frame
            self.call_stack.increment_subcall_index()

            # Reset parser for parent procedure
            self.parser.reset()

    def _handle_backtrack(self, command: Command) -> None:
        """
        Handle BACKTRACK command - backtrack state.

        Args:
            command: The BACKTRACK command with level.
        """
        level = int(command.name)
        self.state.backtrack(level)
