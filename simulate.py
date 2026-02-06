from cdcl_data_gen import CDCLSolver, CDCLEnvironment, MockLLM

# Example SAT problem: (x1 OR x2) AND (NOT x1 OR x3) AND (NOT x2 OR NOT x3)
clauses = [[1, 2], [-1, 3], [-2, -3]]
variables = [1, 2, 3]

# Generate traces with solver
solver = CDCLSolver(clauses, variables)
solver.solve()

# Replay with environment
mock_llm = MockLLM(solver.nested_traces)
env = CDCLEnvironment(clauses, mock_llm)
result = env.solve()  # True=SAT, False=UNSAT

print(f"Result: {'SAT' if result else 'UNSAT'}")