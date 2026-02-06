"""
Trace formatting utilities for CDCL solver traces.

All multi-digit integers are formatted with space-separated digits (e.g., 12 -> "1 2").
This is critical for LLM tokenization.
"""

from typing import Dict, List, Optional


def fmt_int(i: int) -> str:
    """Format integer with space-separated digits: 12 -> '1 2'"""
    return ' '.join(str(i))


def fmt_var(var: int) -> str:
    """Format variable: 5 -> 'x 5'"""
    return f"x {fmt_int(var)}"


def fmt_lit(lit: int) -> str:
    """Format literal: -5 -> '- x 5', 5 -> '+ x 5'"""
    sign = '+' if lit > 0 else '-'
    return f"{sign} x {fmt_int(abs(lit))}"


def fmt_clause(clause: List[int], clause_id: Optional[str] = None) -> str:
    """
    Format clause: [1, -2, 3] -> '( + x 1 - x 2 + x 3 )'
    With ID: '( + x 1 - x 2 + x 3 ) : c 0'
    """
    lits = ' '.join(fmt_lit(lit) for lit in clause)
    result = f"( {lits} )"
    if clause_id is not None:
        result += f" : {clause_id}"
    return result


def fmt_clause_list(clauses: List[List[int]], clause2id: Dict[tuple, str]) -> str:
    """Format list of clauses with their IDs."""
    parts = []
    for clause in clauses:
        cid = clause2id[tuple(clause)]
        parts.append(fmt_clause(clause, cid))
    return ' , '.join(parts)


def fmt_assignments(assignments: Dict[int, bool]) -> str:
    """Format assignments: {1: True, 2: False} -> 'x 1 = True , x 2 = False'"""
    if not assignments:
        return ''
    sorted_items = sorted(assignments.items(), key=lambda x: x[0])
    parts = [f"{fmt_var(var)} = {value}" for var, value in sorted_items]
    return ' , '.join(parts)


def fmt_decision_levels(decision_level: Dict[int, int]) -> str:
    """Format decision levels: {1: 0, 2: 1} -> 'x 1 = 0 , x 2 = 1'"""
    if not decision_level:
        return ''
    parts = [f"{fmt_var(var)} = {fmt_int(level)}" for var, level in decision_level.items()]
    return ' , '.join(parts)


def fmt_reason_clauses(reason_clauses: Dict[int, List[int]], clause2id: Dict[tuple, str]) -> str:
    """Format reason clauses: {var: clause} -> 'x 1 -> c 0 , x 2 -> c 1'"""
    if not reason_clauses:
        return ''
    parts = []
    for var, clause in reason_clauses.items():
        cid = clause2id[tuple(clause)]
        parts.append(f"{fmt_var(var)} -> {cid}")
    return ' , '.join(parts)


def fmt_lit_list(lits: List[int]) -> str:
    """Format list of literals: [1, -2] -> '+ x 1 , - x 2'"""
    if not lits:
        return ''
    return ' , '.join(fmt_lit(lit) for lit in lits)


def fmt_var_list(variables: List[int]) -> str:
    """Format list of variables: [1, 2] -> 'x 1 , x 2'"""
    if not variables:
        return ''
    return ' , '.join(fmt_var(var) for var in variables)


def fmt_level_list(levels: List[int]) -> str:
    """Format list of levels: [0, 1, 2] -> '{ 0 , 1 , 2 }'"""
    if not levels:
        return '{ }'
    inner = ' , '.join(fmt_int(level) for level in levels)
    return f"{{ {inner} }}"
