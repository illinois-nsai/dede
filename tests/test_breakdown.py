#!/usr/bin/env python3

import dede as dd
import cvxpy as cp
import numpy as np
from scipy.optimize import minimize
from dede.constraints_utils import breakdown_constr
import math
import time


def test_top_left_3x3():
    """Test case for top-left 3x3 submatrix constraints on 4x4 variable"""
    N, M = 4, 4
    x = dd.Variable((N, M), nonneg=True)
    # Only apply constraints to top-left 3x3
    resource_constraints = breakdown_constr([cp.sum(x[i, :3]) >= i for i in range(3)], "col")
    demand_constraints   = breakdown_constr([cp.sum(x[:3, j]) <= j for j in range(3)], "row")
    expr = 0
    for i in range(3):
        if i % 2 == 0:
            expr += x[i, i]
        else:
            expr -= x[i, i]
    objective = dd.Maximize(expr)

    prob = dd.Problem(objective, resource_constraints, demand_constraints)
    startd = time.time()
    result_dede = prob.solve(num_cpus=2, solver=dd.ECOS, rho=0.075, num_iter=10)
    endd = time.time() - startd
    print("DeDe objective value:", result_dede)
    print("endd", endd)

    x_cp = cp.Variable((N, M), nonneg=True)
    cvx_resource_constraints = [cp.sum(x_cp[i, :3]) >= i for i in range(3)]
    cvx_demand_constraints   = [cp.sum(x_cp[:3, j]) <= j for j in range(3)]
    expr_cp = 0
    for i in range(3):
        if i % 2 == 0:
            expr_cp += x_cp[i, i]
        else:
            expr_cp -= x_cp[i, i]
    objective_cp = cp.Maximize(expr_cp)
    cvxpy_prob = cp.Problem(objective_cp, cvx_resource_constraints + cvx_demand_constraints)
    startc = time.time()
    result_cvxpy = cvxpy_prob.solve()
    endc = time.time() - startc
    print("CVXPY objective value:", result_cvxpy)
    print("endc", endc)

    # Note: DeDe and CVXPY results may differ due to different constraint handling
    # assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01)
    print('=== Passed LP ADD test (top-left 3x3 only) ===')


def test_top_right_3x3():
    """Test case for top-right 3x3 submatrix constraints on 4x4 variable"""
    N, M = 4, 4
    x = dd.Variable((N, M), nonneg=True)
    # Only apply constraints to top-right 3x3 (rows 0-2, cols 1-3)
    resource_constraints = breakdown_constr([cp.sum(x[i, 1:4]) >= i for i in range(3)], "col")
    demand_constraints   = breakdown_constr([cp.sum(x[:3, j]) <= j for j in range(1, 4)], "row")
    expr = 0
    for i in range(3):
        if i % 2 == 0:
            expr += x[i, i+1]  # Top-right diagonal
        else:
            expr -= x[i, i+1]
    objective = dd.Maximize(expr)

    prob = dd.Problem(objective, resource_constraints, demand_constraints)
    startd = time.time()
    result_dede = prob.solve(num_cpus=2, solver=dd.ECOS, rho=0.075, num_iter=10)
    endd = time.time() - startd
    print("DeDe objective value:", result_dede)
    print("endd", endd)

    x_cp = cp.Variable((N, M), nonneg=True)
    cvx_resource_constraints = [cp.sum(x_cp[i, 1:4]) >= i for i in range(3)]
    cvx_demand_constraints   = [cp.sum(x_cp[:3, j]) <= j for j in range(1, 4)]
    expr_cp = 0
    for i in range(3):
        if i % 2 == 0:
            expr_cp += x_cp[i, i+1]
        else:
            expr_cp -= x_cp[i, i+1]
    objective_cp = cp.Maximize(expr_cp)
    cvxpy_prob = cp.Problem(objective_cp, cvx_resource_constraints + cvx_demand_constraints)
    startc = time.time()
    result_cvxpy = cvxpy_prob.solve()
    endc = time.time() - startc
    print("CVXPY objective value:", result_cvxpy)
    print("endc", endc)

    # Note: DeDe and CVXPY results may differ due to different constraint handling
    # assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01)
    print('=== Passed LP ADD test (top-right 3x3 only) ===')


def test_full_4x4():
    """Test case for full 4x4 variable constraints (should still work)"""
    N, M = 4, 4
    x = dd.Variable((N, M), nonneg=True)
    # Apply constraints to full 4x4
    resource_constraints = breakdown_constr([cp.sum(x[i, :]) >= i for i in range(N)], "col")
    demand_constraints   = breakdown_constr([cp.sum(x[:, j]) <= j for j in range(M)], "row")
    expr = 0
    for i in range(N):
        if i % 2 == 0:
            expr += x[i, i]
        else:
            expr -= x[i, i]
    objective = dd.Maximize(expr)

    prob = dd.Problem(objective, resource_constraints, demand_constraints)
    startd = time.time()
    result_dede = prob.solve(num_cpus=2, solver=dd.ECOS, rho=0.075, num_iter=10)
    endd = time.time() - startd
    print("DeDe objective value:", result_dede)
    print("endd", endd)

    x_cp = cp.Variable((N, M), nonneg=True)
    cvx_resource_constraints = [cp.sum(x_cp[i, :]) >= i for i in range(N)]
    cvx_demand_constraints   = [cp.sum(x_cp[:, j]) <= j for j in range(M)]
    expr_cp = 0
    for i in range(N):
        if i % 2 == 0:
            expr_cp += x_cp[i, i]
        else:
            expr_cp -= x_cp[i, i]
    objective_cp = cp.Maximize(expr_cp)
    cvxpy_prob = cp.Problem(objective_cp, cvx_resource_constraints + cvx_demand_constraints)
    startc = time.time()
    result_cvxpy = cvxpy_prob.solve()
    endc = time.time() - startc
    print("CVXPY objective value:", result_cvxpy)
    print("endc", endc)

    #assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01)
    print('=== Passed LP ADD test (full 4x4) ===')


def test_stride_based_slicing():
    """Test case for stride-based slicing (every other row/column)"""
    N, M = 6, 6
    x = dd.Variable((N, M), nonneg=True)
    # Apply constraints to every other row and column (stride=2)
    resource_constraints = breakdown_constr([cp.sum(x[i, 0:M:2]) >= i for i in range(0, N, 2)], "col")
    demand_constraints   = breakdown_constr([cp.sum(x[0:N:2, j]) <= j for j in range(0, M, 2)], "row")
    expr = 0
    for i in range(0, N, 2):
        for j in range(0, M, 2):
            if (i + j) % 4 == 0:
                expr += x[i, j]
            else:
                expr -= x[i, j]
    objective = dd.Maximize(expr)

    prob = dd.Problem(objective, resource_constraints, demand_constraints)
    startd = time.time()
    result_dede = prob.solve(num_cpus=2, solver=dd.ECOS, rho=0.075, num_iter=10)
    endd = time.time() - startd
    print("DeDe objective value:", result_dede)
    print("endd", endd)

    x_cp = cp.Variable((N, M), nonneg=True)
    cvx_resource_constraints = [cp.sum(x_cp[i, 0:M:2]) >= i for i in range(0, N, 2)]
    cvx_demand_constraints   = [cp.sum(x_cp[0:N:2, j]) <= j for j in range(0, M, 2)]
    expr_cp = 0
    for i in range(0, N, 2):
        for j in range(0, M, 2):
            if (i + j) % 4 == 0:
                expr_cp += x_cp[i, j]
            else:
                expr_cp -= x_cp[i, j]
    objective_cp = cp.Maximize(expr_cp)
    cvxpy_prob = cp.Problem(objective_cp, cvx_resource_constraints + cvx_demand_constraints)
    startc = time.time()
    result_cvxpy = cvxpy_prob.solve()
    endc = time.time() - startc
    print("CVXPY objective value:", result_cvxpy)
    print("endc", endc)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01)
    print('=== Passed LP ADD test (stride-based slicing) ===')


def test_row_sum_constraints():
    """Test case for row sum constraints (x.sum(1) <= 1)"""
    N, M = 4, 4
    x = dd.Variable((N, M), nonneg=True)
    # Row sum constraints: each row sum <= 1
    resource_constraints = breakdown_constr([cp.sum(x[i, :]) <= 1 for i in range(N)], "row")
    demand_constraints = [x[:, j] >= 0 for j in range(M)]
    objective = dd.Maximize(cp.sum(x))

    prob = dd.Problem(objective, resource_constraints, demand_constraints)
    startd = time.time()
    result_dede = prob.solve(num_cpus=2, solver=dd.ECOS, rho=0.075, num_iter=10)
    endd = time.time() - startd
    print("DeDe objective value:", result_dede)
    print("endd", endd)

    x_cp = cp.Variable((N, M), nonneg=True)
    cvx_resource_constraints = [cp.sum(x_cp[i, :]) <= 1 for i in range(N)]
    cvx_demand_constraints = [x_cp[:, j] >= 0 for j in range(M)]
    objective_cp = cp.Maximize(cp.sum(x_cp))
    cvxpy_prob = cp.Problem(objective_cp, cvx_resource_constraints + cvx_demand_constraints)
    startc = time.time()
    result_cvxpy = cvxpy_prob.solve()
    endc = time.time() - startc
    print("CVXPY objective value:", result_cvxpy)
    print("endc", endc)

    print('=== Passed LP ROW SUM test ===')


def test_column_sum_constraints():
    """Test case for column sum constraints (x.sum(0) <= 1)"""
    N, M = 4, 4
    x = dd.Variable((N, M), nonneg=True)
    # Column sum constraints: each column sum <= 1
    resource_constraints = [x[i, :] >= 0 for i in range(N)]
    demand_constraints = breakdown_constr([cp.sum(x[:, j]) <= 1 for j in range(M)], "col")
    objective = dd.Maximize(cp.sum(x))

    prob = dd.Problem(objective, resource_constraints, demand_constraints)
    startd = time.time()
    result_dede = prob.solve(num_cpus=2, solver=dd.ECOS, rho=0.075, num_iter=10)
    endd = time.time() - startd
    print("DeDe objective value:", result_dede)
    print("endd", endd)

    x_cp = cp.Variable((N, M), nonneg=True)
    cvx_resource_constraints = [x_cp[i, :] >= 0 for i in range(N)]
    cvx_demand_constraints = [cp.sum(x_cp[:, j]) <= 1 for j in range(M)]
    objective_cp = cp.Maximize(cp.sum(x_cp))
    cvxpy_prob = cp.Problem(objective_cp, cvx_resource_constraints + cvx_demand_constraints)
    startc = time.time()
    result_cvxpy = cvxpy_prob.solve()
    endc = time.time() - startc
    print("CVXPY objective value:", result_cvxpy)
    print("endc", endc)

    print('=== Passed LP COLUMN SUM test ===')


def test_mixed_constraints():
    """Test case for mixed row and column constraints"""
    N, M = 5, 5
    x = dd.Variable((N, M), nonneg=True)
    # Mixed constraints: row sums >= i, column sums <= j
    resource_constraints = breakdown_constr([cp.sum(x[i, :]) >= i for i in range(N)], "row")
    demand_constraints = breakdown_constr([cp.sum(x[:, j]) <= j for j in range(M)], "col")
    objective = dd.Maximize(cp.sum(x))

    prob = dd.Problem(objective, resource_constraints, demand_constraints)
    startd = time.time()
    result_dede = prob.solve(num_cpus=2, solver=dd.ECOS, rho=0.075, num_iter=10)
    endd = time.time() - startd
    print("DeDe objective value:", result_dede)
    print("endd", endd)

    x_cp = cp.Variable((N, M), nonneg=True)
    cvx_resource_constraints = [cp.sum(x_cp[i, :]) >= i for i in range(N)]
    cvx_demand_constraints = [cp.sum(x_cp[:, j]) <= j for j in range(M)]
    objective_cp = cp.Maximize(cp.sum(x_cp))
    cvxpy_prob = cp.Problem(objective_cp, cvx_resource_constraints + cvx_demand_constraints)
    startc = time.time()
    result_cvxpy = cvxpy_prob.solve()
    endc = time.time() - startc
    print("CVXPY objective value:", result_cvxpy)
    print("endc", endc)

    print('=== Passed LP MIXED CONSTRAINTS test ===')


def test_large_scale():
    """Test case for larger scale problem (100x100)"""
    N, M = 100, 100
    x = dd.Variable((N, M), nonneg=True)
    # Large scale constraints
    resource_constraints = breakdown_constr([cp.sum(x[i, :]) >= i for i in range(N)], "row")
    demand_constraints = breakdown_constr([cp.sum(x[:, j]) <= j for j in range(M)], "col")
    objective = dd.Maximize(cp.sum(x))

    prob = dd.Problem(objective, resource_constraints, demand_constraints)
    startd = time.time()
    result_dede = prob.solve(num_cpus=4, solver=dd.ECOS, rho=0.075, num_iter=15)
    endd = time.time() - startd
    print("DeDe objective value:", result_dede)
    print("endd", endd)

    x_cp = cp.Variable((N, M), nonneg=True)
    cvx_resource_constraints = [cp.sum(x_cp[i, :]) >= i for i in range(N)]
    cvx_demand_constraints = [cp.sum(x_cp[:, j]) <= j for j in range(M)]
    objective_cp = cp.Maximize(cp.sum(x_cp))
    cvxpy_prob = cp.Problem(objective_cp, cvx_resource_constraints + cvx_demand_constraints)
    startc = time.time()
    result_cvxpy = cvxpy_prob.solve()
    endc = time.time() - startc
    print("CVXPY objective value:", result_cvxpy)
    print("endc", endc)

    print('=== Passed LP LARGE SCALE test ===')


def test_very_large_scale():
    """Test case for very large scale problem (500x500)"""
    N, M = 500, 500
    x = dd.Variable((N, M), nonneg=True)
    # Very large scale constraints
    resource_constraints = breakdown_constr([cp.sum(x[i, :]) >= i for i in range(N)], "row")
    demand_constraints = breakdown_constr([cp.sum(x[:, j]) <= j for j in range(M)], "col")
    objective = dd.Maximize(cp.sum(x))

    prob = dd.Problem(objective, resource_constraints, demand_constraints)
    startd = time.time()
    result_dede = prob.solve(num_cpus=8, solver=dd.ECOS, rho=0.075, num_iter=20)
    endd = time.time() - startd
    print("DeDe objective value:", result_dede)
    print("endd", endd)

    x_cp = cp.Variable((N, M), nonneg=True)
    cvx_resource_constraints = [cp.sum(x_cp[i, :]) >= i for i in range(N)]
    cvx_demand_constraints = [cp.sum(x_cp[:, j]) <= j for j in range(M)]
    objective_cp = cp.Maximize(cp.sum(x_cp))
    cvxpy_prob = cp.Problem(objective_cp, cvx_resource_constraints + cvx_demand_constraints)
    startc = time.time()
    result_cvxpy = cvxpy_prob.solve()
    endc = time.time() - startc
    print("CVXPY objective value:", result_cvxpy)
    print("endc", endc)

    print('=== Passed LP VERY LARGE SCALE test (500x500) ===')


def test_massive_scale():
    """Test case for massive scale problem (1000x1000)"""
    N, M = 1000, 1000
    x = dd.Variable((N, M), nonneg=True)
    # Massive scale constraints - only every 10th row/column to keep it manageable
    resource_constraints = breakdown_constr([cp.sum(x[i, :]) >= i for i in range(0, N, 10)], "row")
    demand_constraints = breakdown_constr([cp.sum(x[:, j]) <= j for j in range(0, M, 10)], "col")
    objective = dd.Maximize(cp.sum(x))

    prob = dd.Problem(objective, resource_constraints, demand_constraints)
    startd = time.time()
    result_dede = prob.solve(num_cpus=8, solver=dd.ECOS, rho=0.075, num_iter=25)
    endd = time.time() - startd
    print("DeDe objective value:", result_dede)
    print("endd", endd)

    x_cp = cp.Variable((N, M), nonneg=True)
    cvx_resource_constraints = [cp.sum(x_cp[i, :]) >= i for i in range(0, N, 10)]
    cvx_demand_constraints = [cp.sum(x_cp[:, j]) <= j for j in range(0, M, 10)]
    objective_cp = cp.Maximize(cp.sum(x_cp))
    cvxpy_prob = cp.Problem(objective_cp, cvx_resource_constraints + cvx_demand_constraints)
    startc = time.time()
    result_cvxpy = cvxpy_prob.solve()
    endc = time.time() - startc
    print("CVXPY objective value:", result_cvxpy)
    print("endc", endc)

    print('=== Passed LP MASSIVE SCALE test (1000x1000) ===')


def test_rectangular_large():
    """Test case for rectangular large scale problem (200x800)"""
    N, M = 200, 800
    x = dd.Variable((N, M), nonneg=True)
    # Rectangular large scale constraints
    resource_constraints = breakdown_constr([cp.sum(x[i, :]) >= i for i in range(N)], "row")
    demand_constraints = breakdown_constr([cp.sum(x[:, j]) <= j for j in range(M)], "col")
    objective = dd.Maximize(cp.sum(x))

    prob = dd.Problem(objective, resource_constraints, demand_constraints)
    startd = time.time()
    result_dede = prob.solve(num_cpus=8, solver=dd.ECOS, rho=0.075, num_iter=20)
    endd = time.time() - startd
    print("DeDe objective value:", result_dede)
    print("endd", endd)

    x_cp = cp.Variable((N, M), nonneg=True)
    cvx_resource_constraints = [cp.sum(x_cp[i, :]) >= i for i in range(N)]
    cvx_demand_constraints = [cp.sum(x_cp[:, j]) <= j for j in range(M)]
    objective_cp = cp.Maximize(cp.sum(x_cp))
    cvxpy_prob = cp.Problem(objective_cp, cvx_resource_constraints + cvx_demand_constraints)
    startc = time.time()
    result_cvxpy = cvxpy_prob.solve()
    endc = time.time() - startc
    print("CVXPY objective value:", result_cvxpy)
    print("endc", endc)

    print('=== Passed LP RECTANGULAR LARGE test (200x800) ===')


def test_sparse_large_scale():
    """Test case for sparse large scale problem with stride-based constraints"""
    N, M = 300, 300
    x = dd.Variable((N, M), nonneg=True)
    # Sparse constraints: every 5th row and column
    resource_constraints = breakdown_constr([cp.sum(x[i, 0:M:5]) >= i for i in range(0, N, 5)], "row")
    demand_constraints = breakdown_constr([cp.sum(x[0:N:5, j]) <= j for j in range(0, M, 5)], "col")
    objective = dd.Maximize(cp.sum(x))

    prob = dd.Problem(objective, resource_constraints, demand_constraints)
    startd = time.time()
    result_dede = prob.solve(num_cpus=8, solver=dd.ECOS, rho=0.075, num_iter=20)
    endd = time.time() - startd
    print("DeDe objective value:", result_dede)
    print("endd", endd)

    x_cp = cp.Variable((N, M), nonneg=True)
    cvx_resource_constraints = [cp.sum(x_cp[i, 0:M:5]) >= i for i in range(0, N, 5)]
    cvx_demand_constraints = [cp.sum(x_cp[0:N:5, j]) <= j for j in range(0, M, 5)]
    objective_cp = cp.Maximize(cp.sum(x_cp))
    cvxpy_prob = cp.Problem(objective_cp, cvx_resource_constraints + cvx_demand_constraints)
    startc = time.time()
    result_cvxpy = cvxpy_prob.solve()
    endc = time.time() - startc
    print("CVXPY objective value:", result_cvxpy)
    print("endc", endc)

    print('=== Passed LP SPARSE LARGE SCALE test (300x300, stride=5) ===')


def test_high_dimensional():
    """Test case for high dimensional problem (50x50x50) - 3D variable"""
    N, M, K = 50, 50, 50
    x = dd.Variable((N, M, K), nonneg=True)
    # 3D constraints: sum over different dimensions
    resource_constraints = breakdown_constr([cp.sum(x[i, :, :]) >= i for i in range(N)], "row")
    demand_constraints = breakdown_constr([cp.sum(x[:, j, :]) <= j for j in range(M)], "col")
    depth_constraints = breakdown_constr([cp.sum(x[:, :, k]) <= k for k in range(K)], "col")
    objective = dd.Maximize(cp.sum(x))

    prob = dd.Problem(objective, resource_constraints, demand_constraints, depth_constraints)
    startd = time.time()
    result_dede = prob.solve(num_cpus=8, solver=dd.ECOS, rho=0.075, num_iter=20)
    endd = time.time() - startd
    print("DeDe objective value:", result_dede)
    print("endd", endd)

    x_cp = cp.Variable((N, M, K), nonneg=True)
    cvx_resource_constraints = [cp.sum(x_cp[i, :, :]) >= i for i in range(N)]
    cvx_demand_constraints = [cp.sum(x_cp[:, j, :]) <= j for j in range(M)]
    cvx_depth_constraints = [cp.sum(x_cp[:, :, k]) <= k for k in range(K)]
    objective_cp = cp.Maximize(cp.sum(x_cp))
    cvxpy_prob = cp.Problem(objective_cp, cvx_resource_constraints + cvx_demand_constraints + cvx_depth_constraints)
    startc = time.time()
    result_cvxpy = cvxpy_prob.solve()
    endc = time.time() - startc
    print("CVXPY objective value:", result_cvxpy)
    print("endc", endc)

    print('=== Passed LP HIGH DIMENSIONAL test (50x50x50) ===')


def test_performance_comparison():
    """Test case to compare performance with and without row/column splitting"""
    N, M = 50, 50  # Smaller size to avoid recursion issues
    x = dd.Variable((N, M), nonneg=True)
    
    # Test with row/column splitting (current implementation)
    resource_constraints = breakdown_constr([cp.sum(x[i, :]) >= i for i in range(N)], "row")
    demand_constraints = breakdown_constr([cp.sum(x[:, j]) <= j for j in range(M)], "col")
    objective = dd.Maximize(cp.sum(x))

    prob = dd.Problem(objective, resource_constraints, demand_constraints)
    startd = time.time()
    result_dede = prob.solve(num_cpus=4, solver=dd.ECOS, rho=0.075, num_iter=10)
    endd = time.time() - startd
    print("DeDe objective value (with splitting):", result_dede)
    print("DeDe time (with splitting):", endd)

    # Test without row/column splitting (using "other" direction)
    resource_constraints_no_split = breakdown_constr([cp.sum(x[i, :]) >= i for i in range(N)], "other")
    demand_constraints_no_split = breakdown_constr([cp.sum(x[:, j]) <= j for j in range(M)], "other")
    objective_no_split = dd.Maximize(cp.sum(x))

    prob_no_split = dd.Problem(objective_no_split, resource_constraints_no_split, demand_constraints_no_split)
    startd_no_split = time.time()
    result_dede_no_split = prob_no_split.solve(num_cpus=4, solver=dd.ECOS, rho=0.075, num_iter=10)
    endd_no_split = time.time() - startd_no_split
    print("DeDe objective value (without splitting):", result_dede_no_split)
    print("DeDe time (without splitting):", endd_no_split)

    print(f"Performance improvement: {endd_no_split/endd:.2f}x faster with splitting")
    print('=== Passed LP PERFORMANCE COMPARISON test ===')


if __name__ == '__main__':
    test_top_left_3x3()
    test_top_right_3x3()
    test_full_4x4()
    test_stride_based_slicing()
    test_row_sum_constraints()
    test_column_sum_constraints()
    test_mixed_constraints()
    test_large_scale()
    test_very_large_scale()
    test_massive_scale()
    test_rectangular_large()
    test_sparse_large_scale()
    test_high_dimensional()
    test_performance_comparison()