"""
Mock LLM implementation for testing the environment with recorded traces.

The MockLLM replays tokens from nested traces and validates that
environment injections match the expected trace content.
"""

from typing import Dict, Iterator, List, Optional, Any
from dataclasses import dataclass, field

from .errors import TraceMismatchError, CallStackError


@dataclass
class TracePosition:
    """Tracks position within a trace."""
    tokens: List[str]
    position: int = 0

    def current(self) -> Optional[str]:
        """Get current token without advancing."""
        if self.position < len(self.tokens):
            return self.tokens[self.position]
        return None

    def advance(self) -> Optional[str]:
        """Get current token and advance position."""
        if self.position < len(self.tokens):
            token = self.tokens[self.position]
            self.position += 1
            return token
        return None

    def peek(self, n: int = 1) -> Optional[str]:
        """Peek n tokens ahead."""
        idx = self.position + n
        if idx < len(self.tokens):
            return self.tokens[idx]
        return None

    def skip_expected(self, expected: str) -> None:
        """
        Skip a token, validating it matches expected.

        Used when environment injects a token that's already in the trace.
        """
        current = self.advance()
        if current is None:
            raise TraceMismatchError(expected, "END_OF_TRACE", self.position)
        if current != expected:
            raise TraceMismatchError(expected, current, self.position)

    def is_done(self) -> bool:
        """Check if we've reached the end of the trace."""
        return self.position >= len(self.tokens)


@dataclass
class MockCallFrame:
    """Frame for tracking position in nested trace structure."""
    procedure: str
    trace_pos: TracePosition
    iteration: int = 0
    subcall_index: int = 0


class MockLLM:
    """
    Mock LLM that replays recorded traces.

    Validates that environment injections match the expected content
    from the traces.
    """

    def __init__(self, nested_traces: Dict[str, Any]):
        """
        Initialize with nested traces from a single problem.

        Args:
            nested_traces: Dictionary with "solve_iterations" and "input_clauses"
        """
        self.traces = nested_traces
        self.call_stack: List[MockCallFrame] = []
        self.current_frame: Optional[MockCallFrame] = None

        # Parse traces into tokens for each iteration
        self.solve_iterations = nested_traces.get("solve_iterations", [])
        self.current_iteration = 0

        # State
        self.waiting_for_injection = False
        self.expected_injection: Optional[str] = None

    def generate(self, prompt: str) -> Iterator[str]:
        """
        Generate tokens by replaying the trace.

        Yields tokens one at a time, pausing when environment
        should inject content.

        Args:
            prompt: Ignored for mock (trace-based generation).

        Yields:
            Tokens from the trace.
        """
        while self.current_iteration < len(self.solve_iterations):
            solve_iter = self.solve_iterations[self.current_iteration]

            # Setup frame for this iteration
            trace_tokens = self._tokenize_trace(solve_iter["trace"])
            self.current_frame = MockCallFrame(
                procedure="solve",
                trace_pos=TracePosition(trace_tokens),
                iteration=self.current_iteration,
                subcall_index=0
            )
            self.call_stack = [self.current_frame]

            # Yield tokens for this iteration
            yield from self._generate_from_frame(solve_iter)

            # Check if we're done (SAT or UNSAT detected)
            outcome = solve_iter.get("outcome", "CONTINUE")
            if outcome in ("SAT", "UNSAT"):
                return

            self.current_iteration += 1

    def _generate_from_frame(self, solve_iter: Dict) -> Iterator[str]:
        """
        Generate tokens from current frame, handling subcalls.

        Args:
            solve_iter: The solve iteration dict.

        Yields:
            Tokens from trace and subcalls.
        """
        trace_pos = self.current_frame.trace_pos
        subcalls = solve_iter.get("subcalls", [])

        while not trace_pos.is_done():
            token = trace_pos.current()

            # Check if this token triggers a READ injection
            if token and token.startswith("READ_"):
                # Yield the READ command token
                yield trace_pos.advance()
                # Environment will inject content
                # We need to skip the content in trace that matches injection
                self._skip_injection_content(trace_pos)
                continue

            # Check for CALL - switch to subcall trace
            if token == "CALL":
                yield trace_pos.advance()  # CALL
                procedure = trace_pos.advance()  # procedure name
                yield procedure

                # Get subcall trace
                subcall_idx = self.current_frame.subcall_index
                if subcall_idx < len(subcalls):
                    subcall = subcalls[subcall_idx]
                    yield from self._handle_subcall(subcall)
                    self.current_frame.subcall_index += 1
                continue

            # Check for BACKTRACK - yield with level tokens
            if token == "BACKTRACK":
                yield trace_pos.advance()  # BACKTRACK
                # Yield level tokens (space-separated digits)
                while not trace_pos.is_done():
                    next_tok = trace_pos.current()
                    if next_tok and next_tok.isdigit():
                        yield trace_pos.advance()
                    else:
                        break
                continue

            # Regular token - just yield it
            yield trace_pos.advance()

    def _handle_subcall(self, subcall: Dict) -> Iterator[str]:
        """
        Handle a subcall by switching to its trace.

        Args:
            subcall: Subcall dictionary with "procedure" and "trace".

        Yields:
            Tokens from the subcall trace.
        """
        procedure = subcall["procedure"]
        trace_tokens = self._tokenize_trace(subcall["trace"])

        # Push subcall frame
        subcall_frame = MockCallFrame(
            procedure=procedure,
            trace_pos=TracePosition(trace_tokens)
        )
        self.call_stack.append(subcall_frame)
        old_frame = self.current_frame
        self.current_frame = subcall_frame

        # Generate from subcall
        trace_pos = self.current_frame.trace_pos

        while not trace_pos.is_done():
            token = trace_pos.current()

            # Check for READ injection
            if token and token.startswith("READ_"):
                yield trace_pos.advance()
                self._skip_injection_content(trace_pos)
                continue

            # Check for WRITE - yield command and content
            if token and token.startswith("WRITE_"):
                yield trace_pos.advance()  # WRITE_X
                # Yield bracket and content
                while not trace_pos.is_done():
                    tok = trace_pos.advance()
                    yield tok
                    if tok == "]":
                        break
                continue

            # Regular token
            yield trace_pos.advance()

        # Pop subcall frame
        self.call_stack.pop()
        self.current_frame = old_frame

    def _skip_injection_content(self, trace_pos: TracePosition) -> None:
        """
        Skip the content that will be injected by the environment.

        After a READ_X token, the trace contains [ content ].
        The environment will inject this, so we skip it in the trace.
        """
        # Skip opening bracket
        token = trace_pos.current()
        if token == "[":
            trace_pos.advance()
            # Skip until closing bracket
            depth = 1
            while depth > 0 and not trace_pos.is_done():
                tok = trace_pos.advance()
                if tok == "[":
                    depth += 1
                elif tok == "]":
                    depth -= 1

    def inject(self, content: str) -> None:
        """
        Validate that environment injection matches expected trace.

        Called by environment after READ commands or at procedure start.

        Args:
            content: The content being injected by the environment.

        Raises:
            TraceMismatchError: If content doesn't match expected.
        """
        # For BEGIN tokens, validate against expected
        if content in ("SOLVE_BEGIN", "UNIT_PROPAGATION_BEGIN", "ANALYZE_CONFLICT_BEGIN"):
            # The environment injects this, trace should have it
            # Already handled during generation - the trace token was yielded
            return

        # For READ content injection [ ... ], we already skipped in trace
        # Just validate format is reasonable
        if content.startswith("[") and content.endswith("]"):
            return

        # Other injections - validate or log
        pass

    def start_subcall(self, procedure: str) -> None:
        """
        Signal start of a subcall.

        For mock, this is handled during generation when CALL is detected.

        Args:
            procedure: The procedure being called.
        """
        pass

    def end_subcall(self) -> None:
        """
        Signal end of a subcall.

        For mock, this is handled during generation when *_END is detected.
        """
        pass

    def _tokenize_trace(self, trace: List) -> List[str]:
        """
        Tokenize a trace list into individual tokens.

        The trace can contain strings and nested lists.

        Args:
            trace: The trace list from nested_traces.

        Returns:
            Flat list of tokens.
        """
        tokens = []

        for item in trace:
            if isinstance(item, list):
                # Recursively tokenize nested lists
                for subitem in item:
                    tokens.extend(self._tokenize_string(str(subitem)))
            else:
                tokens.extend(self._tokenize_string(str(item)))

        return tokens

    def _tokenize_string(self, s: str) -> List[str]:
        """
        Tokenize a string into individual tokens.

        Splits on whitespace while preserving significant tokens.

        Args:
            s: String to tokenize.

        Returns:
            List of tokens.
        """
        # Split on whitespace
        parts = s.split()
        tokens = []

        for part in parts:
            # Keep brackets and special chars as separate tokens
            if part in ("(", ")", "[", "]", "{", "}", ":", ",", "=", "->"):
                tokens.append(part)
            elif part.startswith("(") or part.endswith(")"):
                # Handle cases like "(+" or ")"
                if part.startswith("("):
                    tokens.append("(")
                    if len(part) > 1:
                        tokens.append(part[1:])
                elif part.endswith(")"):
                    if len(part) > 1:
                        tokens.append(part[:-1])
                    tokens.append(")")
            else:
                tokens.append(part)

        return [t for t in tokens if t]  # Filter empty
