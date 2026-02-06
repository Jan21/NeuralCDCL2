"""
Custom exceptions for the CDCL environment.
"""


class TraceMismatchError(Exception):
    """Raised when mock LLM detects mismatch between expected and actual content."""

    def __init__(self, expected: str, actual: str, position: int = -1, context: str = ""):
        self.expected = expected
        self.actual = actual
        self.position = position
        self.context = context
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        msg = f"Trace mismatch at position {self.position}:\n"
        msg += f"  Expected: {repr(self.expected)}\n"
        msg += f"  Actual:   {repr(self.actual)}"
        if self.context:
            msg += f"\n  Context:  {self.context}"
        return msg


class InvalidCommandError(Exception):
    """Raised when LLM outputs malformed command."""

    def __init__(self, command: str, reason: str = ""):
        self.command = command
        self.reason = reason
        super().__init__(self._format_message())

    def _format_message(self) -> str:
        msg = f"Invalid command: {repr(self.command)}"
        if self.reason:
            msg += f"\n  Reason: {self.reason}"
        return msg


class UnexpectedEndError(Exception):
    """Raised when trace ends unexpectedly."""

    def __init__(self, procedure: str, position: int = -1):
        self.procedure = procedure
        self.position = position
        super().__init__(f"Unexpected end of trace in {procedure} at position {position}")


class CallStackError(Exception):
    """Raised when call stack operations fail."""

    def __init__(self, message: str):
        super().__init__(message)
