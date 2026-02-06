# LLM-Solver Environment Specification

## Overview

This document specifies an environment that communicates with an LLM to simulate CDCL solver behavior. The environment:
1. Observes LLM token output in a streaming fashion
2. Maintains solver state (assignments, decision levels, learned clauses, etc.)
3. Injects READ data when the LLM requests it
4. Handles nested procedure calls (CALL unit_propagate, CALL analyze_conflict)
5. Can be tested with a mock LLM interface that replays recorded traces

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Environment                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │ State Store │  │ Token Parser│  │ Call Stack Manager      │  │
│  │ - assignments│  │ - streaming │  │ - nested sessions       │  │
│  │ - levels    │  │ - commands  │  │ - context preservation  │  │
│  │ - clauses   │  │             │  │                         │  │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ token stream
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     LLM Interface                                │
│  ┌─────────────────────┐    ┌─────────────────────────────────┐ │
│  │   Real LLM API      │ OR │   Mock LLM (Trace Replay)       │ │
│  │   (future)          │    │   - validates env injections    │ │
│  │                     │    │   - yields tokens from trace    │ │
│  └─────────────────────┘    └─────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

## State Store

The environment maintains the following solver state:

```python
class SolverState:
    clauses: List[List[int]]           # Input + learned clauses
    assignments: Dict[int, bool]        # var -> value
    decision_level: Dict[int, int]      # var -> level assigned
    reason_clauses: Dict[int, List[int]] # var -> clause that implied it
    learned_clauses: List[List[int]]    # Learned clauses only
    level: int                          # Current decision level
    clause2id: Dict[Tuple[int], str]    # clause -> "c 0", "c 1", etc.
```

## Token Parsing

### Streaming Mode

The environment processes tokens one at a time as they arrive from the LLM. It maintains a buffer to detect command patterns.

### Command Detection

Commands are detected incrementally:

| Command Type | Pattern | Action |
|-------------|---------|--------|
| READ_X | `READ_ASSIGNMENTS`, `READ_CLAUSES`, etc. | Environment injects `[ content ]` |
| WRITE_X | `WRITE_ASSIGNMENTS [ content ]` | Environment parses content, updates state |
| CALL | `CALL unit_propagate`, `CALL analyze_conflict` | Push call stack, prepare new prompt |
| *_END | `UNIT_PROPAGATION_END`, `ANALYZE_CONFLICT_END`, `SOLVE_END` | Pop call stack if in subcall |
| BACKTRACK | `BACKTRACK <level>` | Environment backtracks state |

### Token Buffer

```python
class TokenBuffer:
    tokens: List[str]

    def add(self, token: str) -> Optional[Command]:
        """Add token, return Command if pattern complete."""
```

## Command Processing

### READ Commands

When the LLM outputs a READ command token (e.g., `READ_ASSIGNMENTS`), the environment immediately injects the data:

```
LLM outputs:  READ_ASSIGNMENTS
Env injects:  [ x 1 = True , x 2 = False ]
LLM continues: (next command)
```

Supported READ commands:
- `READ_ASSIGNMENTS` → current variable assignments
- `READ_CLAUSES` → all clauses (input + learned)
- `READ_DECISION_LEVELS` → variable decision levels
- `READ_LEVEL` → current decision level
- `READ_CONFLICT_CLAUSE` → clause that caused conflict (if any)
- `READ_REASON_CLAUSES` → reason clauses for propagated variables
- `READ_LEARNED_CLAUSE` → most recently learned clause

### WRITE Commands

The LLM outputs WRITE commands with content. The environment parses and validates:

```
LLM outputs:  WRITE_ASSIGNMENTS [ x 3 = True BECAUSE c 1 ]
Env action:   Parse, validate format, update state.assignments[3] = True
```

Supported WRITE commands:
- `WRITE_ASSIGNMENTS [ x <var> = <True|False> BECAUSE <clause_id> ]` for this command there could be or not the BECAUSE <clause_id>. If it is there the environment will parse it as a reason_clause
- `WRITE_CONFLICT_CLAUSE [ <clause_id> ]`
- `WRITE_LEARNED_CLAUSE [ ( +/- x <var> ... ) ]`

### CALL Commands

When LLM outputs `CALL <procedure>`:

1. Environment pushes current context to call stack
2. Environment prepares initial prompt for subcall:
   - `CALL unit_propagate` → inject `UNIT_PROPAGATION_BEGIN`
   - `CALL analyze_conflict` → inject `ANALYZE_CONFLICT_BEGIN`
3. LLM generates subcall trace
4. When `*_END` pattern detected, pop call stack and resume parent, when it is SOLVE_ITERATION_END then it will go to next iteration

### BACKTRACK Command

```
LLM outputs:  BACKTRACK 1
Env action:
  - Set level = 1
  - Remove assignments where decision_level[var] > 1
  - Remove corresponding reason_clauses entries
```

## Call Stack Management

```python
class CallStackFrame:
    procedure: str           # "solve", "unit_propagate", "analyze_conflict"
    parent_state: dict       # Snapshot of environment state
    trace_position: int      # For mock: position in trace

class CallStack:
    frames: List[CallStackFrame]

    def push(self, procedure: str):
        """Push new frame, save parent context."""

    def pop(self) -> CallStackFrame:
        """Pop frame, restore parent context."""
```

## Mock LLM Interface

For testing without a real LLM, the mock interface replays recorded traces.

### Interface Contract

```python
class LLMInterface(Protocol):
    def generate(self, prompt: str) -> Iterator[str]:
        """Yield tokens one at a time."""

    def inject(self, content: str):
        """Environment injects content (for READ responses)."""
```

### Mock Implementation

```python
class MockLLM:
    def __init__(self, nested_traces: dict):
        self.traces = nested_traces
        self.current_trace: List[str]
        self.position: int
        self.expected_injections: Queue

    def generate(self, prompt: str) -> Iterator[str]:
        """
        Yield tokens from trace.
        When environment should inject, pause and wait.
        """

    def inject(self, content: str):
        """
        Validate that injected content matches expected trace.
        Raises exception if mismatch.
        """
```

### Validation Rules

The mock interface validates:

1. **Initial tokens**: When environment provides `SOLVE_BEGIN`, `UNIT_PROPAGATION_BEGIN`, etc., verify it matches the trace.

2. **READ injections**: When environment injects `[ x 1 = True ]` after `READ_ASSIGNMENTS`, verify it matches what's in the trace.

3. **WRITE outputs**: When mock yields `WRITE_ASSIGNMENTS [ ... ]`, environment will process it. Mock verifies LLM output matches trace.

4. **Strict mode**: Any mismatch raises `TraceMismatchError` immediately.

### Trace Navigation

The mock must navigate nested traces:

```python
nested_traces = {
    "input_clauses": {...},
    "solve_iterations": [
        {
            "procedure": "solve",
            "trace": [...],
            "subcalls": [
                {"procedure": "unit_propagate", "trace": [...], "subcalls": []},
                {"procedure": "analyze_conflict", "trace": [...], "subcalls": []}
            ],
            "state_before": {...},
            "state_after": {...}
        }
    ]
}
```

When `CALL unit_propagate` is reached:
1. Mock pauses current trace position
2. Switches to corresponding subcall trace
3. After `UNIT_PROPAGATION_END`, returns to parent trace

## Environment API

```python
class CDCLEnvironment:
    def __init__(self, clauses: List[List[int]], llm: LLMInterface):
        self.state = SolverState(clauses)
        self.llm = llm
        self.call_stack = CallStack()
        self.parser = TokenParser()

    def solve(self) -> bool:
        """
        Run the solver using LLM for decisions.
        Returns SAT/UNSAT result.
        """
        # Inject SOLVE_BEGIN
        self.llm.inject("SOLVE_BEGIN")

        for token in self.llm.generate(self._build_prompt()):
            command = self.parser.add(token)
            if command:
                self._handle_command(command)

            if self._is_terminal():
                break

        return self.state.result

    def _handle_command(self, command: Command):
        match command.type:
            case "READ":
                content = self._get_read_content(command.name)
                self.llm.inject(f"[ {content} ]")
            case "WRITE":
                self._process_write(command.name, command.content)
            case "CALL":
                self._handle_call(command.procedure)
            case "END":
                self._handle_end(command.procedure)
            case "BACKTRACK":
                self._backtrack(command.level)
```

## Token Format

All multi-digit numbers use space-separated digits:
- `12` → `1 2`
- Variable 12: `x 1 2`
- Clause ID 12: `c 1 2`

## Example Flow

### Solve Iteration with Unit Propagation

```
1. Environment injects: SOLVE_BEGIN
2. LLM yields: CALL unit_propagate
3. Environment pushes call stack, injects: UNIT_PROPAGATION_BEGIN
4. LLM yields: READ_ASSIGNMENTS
5. Environment injects: [ x 1 = True , x 2 = False ]
6. LLM yields: READ_CLAUSES
7. Environment injects: [ ( + x 1 - x 2 ) : c 0 , ... ]
8. LLM yields: EVALUATE_CLAUSE ( + x 1 - x 2 ) : c 0
9. LLM yields: PROPAGATE
10. LLM yields: WRITE_ASSIGNMENTS [ x 3 = True BECAUSE c 1 ]
11. Environment updates state.assignments[3] = True
12. LLM yields: UNIT_PROPAGATION_END
13. Environment pops call stack, resumes solve
14. LLM yields: READ_ASSIGNMENTS
15. Environment injects: [ x 1 = True , x 2 = False , x 3 = True ]
16. ... (continues)
```

## Error Handling

### TraceMismatchError

Raised when mock LLM detects mismatch:
- Environment injection doesn't match expected trace content
- LLM output (from trace) doesn't match expected pattern

```python
class TraceMismatchError(Exception):
    def __init__(self, expected: str, actual: str, position: int):
        self.expected = expected
        self.actual = actual
        self.position = position
```

### InvalidCommandError

Raised when LLM outputs malformed command:
- Unknown command type
- Malformed content (e.g., invalid variable format)

## File Structure

```
cdcl_data_gen/
├── environment.py      # CDCLEnvironment, SolverState
├── parser.py           # TokenParser, TokenBuffer
├── call_stack.py       # CallStack, CallStackFrame
├── mock_llm.py         # MockLLM implementation
├── llm_interface.py    # LLMInterface protocol
└── errors.py           # TraceMismatchError, InvalidCommandError
```

## Testing Strategy

1. **Unit tests**: Test each component (parser, state store, call stack)
2. **Integration tests**: Run mock LLM with various traces
3. **Verification**: Compare final state with `state_after` from nested traces
4. **Edge cases**:
   - Empty assignments
   - Immediate conflicts (level 0)
   - Deep nesting (multiple conflicts)

## Future Extensions

1. **Real LLM integration**: Replace MockLLM with API calls to actual LLM
2. **Streaming API**: Support OpenAI/Anthropic streaming APIs
3. **Partial generation**: Handle cases where LLM makes mistakes
4. **Training data collection**: Record successful environment-LLM interactions
