"""
Case 4a: from_cone treats the term as a constant ([] → group 0); from_tree
finds the position in some constraint group and assigns the term there.
Both succeed but assign the term to DIFFERENT groups. Since the term contributes
0, the answer is still correct — only load balancing differs.

  Variable x of shape (2,)
  Coefficient w = [1, 0]
  Constraints reference both x[0] and x[1]    (x[1] is NOT unconstrained)
  Objective: sum(multiply(w, x))  →  expand_expr splits into per-element terms

For the second term — Sum(multiply(0, x[1])):
  - from_cone:  returns []                     → falls into `if not var_id_pos_list`
                                                 branch → assigned to group 0 unconditionally
  - from_tree:  returns [(x_id, 1)]            → looks up var_id_pos_to_idx,
                                                 finds the group containing x[1],
                                                 assigns the term there

Both solves succeed; the term contributes 0 either way, so the optimal value
is identical. The only observable difference is which worker "owns" the term.
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

# x[1] IS in constraints this time (contrast with Case 1)
resource_constraints = [x[0] >= 1, x[1] >= 1]
demand_constraints = [x[0] <= 10, x[1] <= 10]

objective = dd.Minimize(dd.sum(dd.multiply(w, x)))

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
