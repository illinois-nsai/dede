import cvxpy as cp
from dede.constraints_utils import breakdown_expression

N, M = 4, 3
x = cp.Variable((N, M))

# Test Variable breakdown by row
row_groups = breakdown_expression(x, dir="row")
assert len(row_groups) == N and all(g.shape == (1, M) or g.shape == (M,) for g in row_groups)
print("Row slicing for Variable:", [g.shape for g in row_groups])

# Test Variable breakdown by column
col_groups = breakdown_expression(x, dir="col")
assert len(col_groups) == M and all(g.shape == (N, 1) or g.shape == (N,) for g in col_groups)
print("Col slicing for Variable:", [g.shape for g in col_groups])

# 1D Variable remains a single group
v = cp.Variable(7)
assert breakdown_expression(v, dir="row") == [v]
print("1D Variable returns only one group.")
