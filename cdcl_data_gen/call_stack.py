"""
Call stack management for nested procedure calls.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .errors import CallStackError


@dataclass
class CallStackFrame:
    """
    Represents a single frame in the call stack.

    Attributes:
        procedure: Name of the procedure ("solve", "unit_propagate", "analyze_conflict")
        iteration: For solve, the iteration index
        subcall_index: Index into the subcalls array for mock navigation
        trace_position: Position in the current trace (for mock)
        context: Additional context data preserved across the call
    """
    procedure: str
    iteration: int = 0
    subcall_index: int = 0
    trace_position: int = 0
    context: Dict[str, Any] = field(default_factory=dict)


class CallStack:
    """
    Manages the call stack for nested procedure calls.

    The call stack tracks:
    - Current procedure (solve, unit_propagate, analyze_conflict)
    - For mock LLM: position in nested trace structure
    - Context that needs to be preserved across calls
    """

    def __init__(self):
        self.frames: List[CallStackFrame] = []

    def push(self, procedure: str, **context) -> None:
        """
        Push a new frame onto the call stack.

        Args:
            procedure: Name of the procedure being called.
            **context: Additional context to store in the frame.
        """
        frame = CallStackFrame(
            procedure=procedure,
            context=dict(context)
        )
        self.frames.append(frame)

    def pop(self) -> CallStackFrame:
        """
        Pop the top frame from the call stack.

        Returns:
            The popped CallStackFrame.

        Raises:
            CallStackError: If the stack is empty.
        """
        if not self.frames:
            raise CallStackError("Cannot pop from empty call stack")
        return self.frames.pop()

    def peek(self) -> Optional[CallStackFrame]:
        """
        Get the top frame without removing it.

        Returns:
            The top CallStackFrame, or None if empty.
        """
        return self.frames[-1] if self.frames else None

    def current_procedure(self) -> Optional[str]:
        """
        Get the name of the current procedure.

        Returns:
            Procedure name, or None if stack is empty.
        """
        frame = self.peek()
        return frame.procedure if frame else None

    def depth(self) -> int:
        """
        Get the current stack depth.

        Returns:
            Number of frames on the stack.
        """
        return len(self.frames)

    def is_empty(self) -> bool:
        """
        Check if the call stack is empty.

        Returns:
            True if empty, False otherwise.
        """
        return len(self.frames) == 0

    def is_in_subcall(self) -> bool:
        """
        Check if we're in a subcall (not at solve level).

        Returns:
            True if in unit_propagate or analyze_conflict.
        """
        if self.is_empty():
            return False
        return self.current_procedure() in ("unit_propagate", "analyze_conflict")

    def update_trace_position(self, position: int) -> None:
        """
        Update the trace position in the current frame.

        Args:
            position: New position in the trace.
        """
        if self.frames:
            self.frames[-1].trace_position = position

    def increment_subcall_index(self) -> None:
        """
        Increment the subcall index in the current frame.

        Called after completing a subcall to move to the next one.
        """
        if self.frames:
            self.frames[-1].subcall_index += 1

    def get_subcall_index(self) -> int:
        """
        Get the current subcall index.

        Returns:
            Subcall index, or 0 if stack is empty.
        """
        frame = self.peek()
        return frame.subcall_index if frame else 0

    def set_iteration(self, iteration: int) -> None:
        """
        Set the iteration index for the current frame.

        Args:
            iteration: The iteration index.
        """
        if self.frames:
            self.frames[-1].iteration = iteration

    def get_iteration(self) -> int:
        """
        Get the current iteration index.

        Returns:
            Iteration index, or 0 if stack is empty.
        """
        frame = self.peek()
        return frame.iteration if frame else 0

    def clear(self) -> None:
        """Clear all frames from the stack."""
        self.frames = []

    def __repr__(self) -> str:
        if not self.frames:
            return "CallStack(empty)"
        procedures = " -> ".join(f.procedure for f in self.frames)
        return f"CallStack({procedures})"
