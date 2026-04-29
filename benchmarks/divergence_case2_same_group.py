"""
Case 2: Ghost position is in the SAME constraint group as the real position.

To get a single expanded term referencing multiple positions we need an atom
that expand_expr does NOT split — cp.norm fits (no branch in expand_expr → falls
through to `return [expr]`).

  Variable x of shape (2,)
  Coefficient w = [1, 0]
  Single resource constraint x[0] + x[1] >= 1   (puts both in resource group 0)
  Single demand   constraint x[0] + x[1] <= 10  (puts both in demand   group 0)
  Objective: norm(multiply(w, x))  →  one term, references both x[0] and x[1]

For the term — Pnorm(multiply([1, 0], x), 2):
  - from_cone:  returns [(x_id, 0)]                    → intersection = {(0,0), (1,0)}
  - from_tree:  returns [(x_id, 0), (x_id, 1)]         → intersection = {(0,0), (1,0)}
  Both pick the same group. Solve succeeds with identical assignment.
"""

import cvxpy as cp
import numpy as np

import dede as dd
from dede.utils import (
    expand_expr,
    get_var_id_pos_list_from_cone,
    get_var_id_pos_list_from_tree,
)

x = dd.Variable(2, nonneg=True)
w = np.array([1.0, 0.0])

# both positions in the same resource and demand groups
resource_constraints = [x[0] + x[1] >= 1]
demand_constraints = [x[0] + x[1] <= 10]

objective = dd.Minimize(cp.norm(cp.multiply(w, x), 2))

print("Per-term comparison:")
for i, term in enumerate(expand_expr(objective.expr)):
    print(f"  term[{i}] = {term}")
    print(f"    from_tree: {get_var_id_pos_list_from_tree(term)}")
    print(f"    from_cone: {get_var_id_pos_list_from_cone(term, solver='ECOS')}")

print("\nSolving with from_tree (current code path):")
prob = dd.Problem(objective, resource_constraints, demand_constraints)
try:
    result = prob.solve(num_cpus=2, ray_address="auto", solver=dd.ECOS, num_iter=20)
    print(f"  → solved successfully, objective = {result}")
except Exception as e:
    print(f"  → {type(e).__name__}: {e}")
