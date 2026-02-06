"""
Token parser for the CDCL environment.

Processes tokens streaming from the LLM and detects commands.
"""

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional

from .errors import InvalidCommandError


class CommandType(Enum):
    """Types of commands that can be detected."""
    READ = auto()
    WRITE = auto()
    CALL = auto()
    END = auto()
    BACKTRACK = auto()
    BEGIN = auto()
    OTHER = auto()


@dataclass
class Command:
    """Represents a parsed command from the LLM output."""
    type: CommandType
    name: str  # e.g., "ASSIGNMENTS", "CLAUSES", "unit_propagate"
    content: str = ""  # For WRITE commands, the content inside [ ]
    raw: str = ""  # The raw token(s) that formed this command


# Commands that trigger environment actions
READ_COMMANDS = {
    "READ_ASSIGNMENTS",
    "READ_CLAUSES",
    "READ_DECISION_LEVELS",
    "READ_LEVEL",
    "READ_CONFLICT_CLAUSE",
    "READ_REASON_CLAUSES",
    "READ_LEARNED_CLAUSE",
}

WRITE_COMMANDS = {
    "WRITE_ASSIGNMENTS",
    "WRITE_CONFLICT_CLAUSE",
    "WRITE_LEARNED_CLAUSE",
}

END_COMMANDS = {
    "UNIT_PROPAGATION_END",
    "ANALYZE_CONFLICT_END",
    "SOLVE_END",
    "SOLVE_ITER_END",
}

BEGIN_COMMANDS = {
    "SOLVE_BEGIN",
    "UNIT_PROPAGATION_BEGIN",
    "ANALYZE_CONFLICT_BEGIN",
}

CALL_PATTERN = re.compile(r"CALL\s+(unit_propagate|analyze_conflict)")
BACKTRACK_PATTERN = re.compile(r"BACKTRACK\s+([\d\s]+)")


class TokenParser:
    """
    Streaming token parser that detects commands.

    Processes tokens one at a time and returns Command objects
    when complete command patterns are detected.
    """

    def __init__(self):
        self.buffer: List[str] = []
        self.in_bracket = False
        self.bracket_content: List[str] = []
        self.current_write_cmd: Optional[str] = None

    def reset(self):
        """Reset parser state."""
        self.buffer = []
        self.in_bracket = False
        self.bracket_content = []
        self.current_write_cmd = None

    def add_token(self, token: str) -> Optional[Command]:
        """
        Add a token and return a Command if a complete pattern is detected.

        Args:
            token: A single token from the LLM.

        Returns:
            Command if a complete command is detected, None otherwise.
        """
        token = token.strip()
        if not token:
            return None

        # Handle bracket content for WRITE commands
        if self.in_bracket:
            if token == "]":
                # End of bracket content
                self.in_bracket = False
                content = " ".join(self.bracket_content)
                cmd_name = self.current_write_cmd
                self.bracket_content = []
                self.current_write_cmd = None
                return Command(
                    type=CommandType.WRITE,
                    name=cmd_name.replace("WRITE_", ""),
                    content=content,
                    raw=f"{cmd_name} [ {content} ]"
                )
            else:
                self.bracket_content.append(token)
                return None

        # Check for WRITE command start
        if token in WRITE_COMMANDS:
            self.current_write_cmd = token
            return None

        # Check for bracket start after WRITE command
        if token == "[" and self.current_write_cmd:
            self.in_bracket = True
            return None

        # If we had a WRITE command but no bracket, reset
        if self.current_write_cmd and token != "[":
            # This shouldn't happen in valid traces
            self.current_write_cmd = None

        # Check for READ commands
        if token in READ_COMMANDS:
            return Command(
                type=CommandType.READ,
                name=token.replace("READ_", ""),
                raw=token
            )

        # Check for END commands
        if token in END_COMMANDS:
            # Extract procedure name from end command
            name = token.replace("_END", "").lower()
            return Command(
                type=CommandType.END,
                name=name,
                raw=token
            )

        # Check for BEGIN commands
        if token in BEGIN_COMMANDS:
            name = token.replace("_BEGIN", "").lower()
            return Command(
                type=CommandType.BEGIN,
                name=name,
                raw=token
            )

        # Check for CALL command - need to buffer for procedure name
        if token == "CALL":
            self.buffer = ["CALL"]
            return None

        # Check if we're completing a CALL command
        if self.buffer and self.buffer[0] == "CALL":
            procedure = token
            self.buffer = []
            if procedure in ("unit_propagate", "analyze_conflict"):
                return Command(
                    type=CommandType.CALL,
                    name=procedure,
                    raw=f"CALL {procedure}"
                )
            else:
                raise InvalidCommandError(
                    f"CALL {procedure}",
                    f"Unknown procedure: {procedure}"
                )

        # Check for BACKTRACK command
        if token == "BACKTRACK":
            self.buffer = ["BACKTRACK"]
            return None

        # Check if we're completing a BACKTRACK command
        if self.buffer and self.buffer[0] == "BACKTRACK":
            # Accumulate digits (space-separated)
            self.buffer.append(token)
            # Try to parse as level
            level_str = " ".join(self.buffer[1:])
            # Check if we have a complete level (next token would be a command)
            # For simplicity, assume single-token level for now
            # In practice, multi-digit levels are space-separated
            try:
                level = int(level_str.replace(" ", ""))
                self.buffer = []
                return Command(
                    type=CommandType.BACKTRACK,
                    name=str(level),
                    raw=f"BACKTRACK {level_str}"
                )
            except ValueError:
                # Not a complete level yet, keep buffering
                return None

        # Other tokens (PROPAGATE, EVALUATE_CLAUSE, etc.) - pass through
        return Command(
            type=CommandType.OTHER,
            name=token,
            raw=token
        )


def parse_write_assignments(content: str) -> dict:
    """
    Parse WRITE_ASSIGNMENTS content.

    Format: x <var> = <True|False> [BECAUSE <clause_id>]

    Returns:
        Dictionary with 'var', 'value', and optionally 'reason'.
    """
    result = {}

    # Pattern: x <spaced_digits> = True|False [BECAUSE c <spaced_digits>]
    pattern = r"x\s+([\d\s]+)\s*=\s*(True|False)(?:\s+BECAUSE\s+(c\s*[\d\s]+))?"
    match = re.search(pattern, content)

    if match:
        # Parse variable (space-separated digits)
        var_str = match.group(1)
        result["var"] = int(var_str.replace(" ", ""))
        result["value"] = match.group(2) == "True"

        # Parse reason clause if present
        if match.group(3):
            result["reason"] = match.group(3).strip()

    return result


def parse_write_conflict_clause(content: str) -> str:
    """
    Parse WRITE_CONFLICT_CLAUSE content.

    Format: c <id>

    Returns:
        Clause ID string.
    """
    return content.strip()


def parse_write_learned_clause(content: str) -> List[int]:
    """
    Parse WRITE_LEARNED_CLAUSE content.

    Format: ( +/- x <var> ... )

    Returns:
        List of literals.
    """
    literals = []
    pattern = r"([+-])\s*x\s*([\d\s]+)"
    for match in re.finditer(pattern, content):
        sign = 1 if match.group(1) == "+" else -1
        var = int(match.group(2).replace(" ", ""))
        literals.append(sign * var)
    return literals


def parse_backtrack_level(content: str) -> int:
    """
    Parse BACKTRACK level.

    Format: space-separated digits

    Returns:
        Integer level.
    """
    return int(content.replace(" ", ""))
