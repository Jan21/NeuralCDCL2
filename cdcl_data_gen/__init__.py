"""
CDCL Data Generation Package

This package provides tools for generating training data for LLMs
that imitate CDCL SAT solver behavior, and an environment for
running an LLM as a solver.
"""

from .solver import CDCLSolver
from .formula import generate_random_formula
from .verifier import TraceVerifier, verify_solver_traces
from .collector import collect_traces, create_datasets
from .format import (
    fmt_int, fmt_var, fmt_lit, fmt_clause, fmt_clause_list,
    fmt_assignments, fmt_decision_levels, fmt_reason_clauses,
    fmt_lit_list, fmt_var_list, fmt_level_list
)

# Environment components
from .environment import CDCLEnvironment, SolverState
from .mock_llm import MockLLM
from .parser import TokenParser, Command, CommandType
from .call_stack import CallStack, CallStackFrame
from .llm_interface import LLMInterface
from .errors import TraceMismatchError, InvalidCommandError, CallStackError

__all__ = [
    # Solver
    'CDCLSolver',

    # Formula generation
    'generate_random_formula',

    # Verification
    'TraceVerifier',
    'verify_solver_traces',

    # Data collection
    'collect_traces',
    'create_datasets',

    # Formatting
    'fmt_int',
    'fmt_var',
    'fmt_lit',
    'fmt_clause',
    'fmt_clause_list',
    'fmt_assignments',
    'fmt_decision_levels',
    'fmt_reason_clauses',
    'fmt_lit_list',
    'fmt_var_list',
    'fmt_level_list',

    # Environment
    'CDCLEnvironment',
    'SolverState',
    'MockLLM',
    'TokenParser',
    'Command',
    'CommandType',
    'CallStack',
    'CallStackFrame',
    'LLMInterface',

    # Errors
    'TraceMismatchError',
    'InvalidCommandError',
    'CallStackError',
]
