"""
Case 1: Ghost position is in NO constraint.

  Variable x of shape (2,)
  Coefficient w = [1, 0]
  Constraints reference only x[0]; x[1] is in no constraint
  Objective: sum(multiply(w, x))  →  expand_expr splits into per-element terms

For the second term — Sum(multiply(0, x[1])) — from_tree returns [(x_id, 1)]
but var_id_pos_to_idx has no entry for that key (built from constraints only).

Outcome:
  - from_cone:  returns [] → assigned to group 0 as constant → solve succeeds
  - from_tree:  KeyError on var_id_pos_to_idx[(x_id, 1)]
                (would be ValueError("not separable") if the dict still had
                 defaultdict behavior after dict() + ray.put)
"""

import numpy as np

import dede as dd
from dede.utils import (
    expand_expr,
    get_var_id_pos_list_from_cone,
    get_var_id_pos_list_from_tree,
)

x = dd.Variable(2, nonneg=True)
w = np.array([1.0, 0.0])

resource_constraints = [x[0] >= 1]
demand_constraints = [x[0] <= 10]

objective = dd.Minimize(dd.sum(dd.multiply(w, x)))

print("Per-term comparison:")
for i, term in enumerate(expand_expr(objective.expr)):
    print(f"  term[{i}] = {term}")
    print(f"    from_tree: {get_var_id_pos_list_from_tree(term)}")
    print(f"    from_cone: {get_var_id_pos_list_from_cone(term, solver='ECOS')}")

print("\nSolving with from_tree (current code path):")
prob = dd.Problem(objective, resource_constraints, demand_constraints)
try:
    prob.solve(num_cpus=2, ray_address="auto", solver=dd.ECOS)
    print("  → solved successfully (unexpected)")
except Exception as e:
    print(f"  → {type(e).__name__}: {e}")
