#!/usr/bin/env python3

import dede as dd
import cvxpy as cp
import numpy as np
from scipy.optimize import minimize
import math


def test_log():
    N, M = 10, 10
    x = dd.Variable((N, M), nonneg=True)

    # nonnegative log values: x >= 1
    resource_constraints = [x >= 1] + [x[i, :].sum() <= (i + 1) * M for i in range(N)]
    demand_constraints = [x[:, j].sum() <= (j + 1) * N for j in range(M)]
    
    # write in separable form, dd.sum(x) fails
    expr = dd.sum([dd.sum(dd.log(x[i, :])) for i in range(N)])
    objective = dd.Maximize(expr)

    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=2, solver=dd.ECOS, rho=0.08, num_iter=30)
    print("DeDe:", result_dede)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve()
    print("CVXPY:", result_cvxpy)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01)
    print('=== Passed CONTINUOUS LOG test ===') 


def test_log_weighted():
    N, M = 10, 10
    x = dd.Variable((N, M), nonneg=True)

    # nonnegative log values: x >= 1
    resource_constraints = [x >= 1] + [x[i, :].sum() <= (i + 1) * M for i in range(N)]
    demand_constraints = [x[:, j].sum() <= (j + 1) * N for j in range(M)]
    
    # write in separable form, dd.sum(x) fails
    w = np.empty((N, M))
    for i in range(N):
        for j in range(M):
            w[i][j] = i + j
    expr = dd.sum(dd.multiply(w, dd.log(x)))
    objective = dd.Maximize(expr)

    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=2, solver=dd.ECOS, rho=1, num_iter=50)
    print("DeDe:", result_dede)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve()
    print("CVXPY:", result_cvxpy)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01)
    print('=== Passed CONTINUOUS LOG weighted test ===')  


def test_quadratic():
    N, M = 10, 10
    x = dd.Variable((N, M), nonneg=True)

    resource_constraints = [x[i, :].sum() >= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= j for j in range(M)]
    objective = dd.Minimize(dd.quad_over_lin(x, 1))
    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=2, solver=dd.ECOS, rho=12, num_iter=7)
    print("DeDe:", result_dede)
    
    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve(solver=cp.ECOS_BB)
    print("CVXPY:", result_cvxpy)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01)
    print('=== Passed CONTINUOUS QUADRATIC test ===')


def test_quadratic_weighted():
    N, M = 10, 10
    x = dd.Variable((N, M), nonneg=True)

    resource_constraints = [x[i, :].sum() >= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= j for j in range(M)]
    w = np.empty((N, M))
    for i in range(N):
        for j in range(M):
            w[i][j] = i * j / (i + j + 1)
    objective = dd.Minimize(dd.quad_over_lin(dd.multiply(x, w), 1))
    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=2, solver=dd.ECOS, rho=20, num_iter=15)
    print("DeDe:", result_dede)
    
    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve(solver=cp.ECOS_BB)
    print("CVXPY:", result_cvxpy)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01)
    print('=== Passed CONTINUOUS QUADRATIC weighted test ===')


if __name__ == '__main__':
    test_log()
    test_log_weighted()

    test_quadratic()
    test_quadratic_weighted()