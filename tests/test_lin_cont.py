#!/usr/bin/env python3

import dede as dd
import cvxpy as cp
import numpy as np
from scipy.optimize import minimize
import math


def test_add1():
    N, M = 10, 10
    x = dd.Variable((N, M), nonneg=True)
    resource_constraints = [x[i, :].sum() >= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= j for j in range(M)]
    expr = 0
    for i in range(min(N, M)):
        if i % 2 == 0:
            expr += x[i, i]
        else:
            expr -= x[i, i]
    objective = dd.Maximize(expr)

    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=2, solver=dd.ECOS, rho=0.1, num_iter=15)
    print("DeDe:", result_dede)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve()
    print("CVXPY:", result_cvxpy)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01)
    print('=== Passed LP ADD test #1 ===')


def test_add2():
    N, M = 10, 10
    x = dd.Variable((N, M), nonneg=True)
    resource_constraints = [x[i, :].sum() <= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= j for j in range(M)]
    
    # pull two elements
    objective = dd.Maximize(x[N-1, M-1] + x[N//2, M//2])

    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=2, solver=dd.ECOS, rho=0.1, num_iter=10)
    print("DeDe:", result_dede)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve()
    print("CVXPY:", result_cvxpy)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01)
    print('=== Passed LP ADD test #2 ===')


def test_add_zero():
    N, M = 3, 3
    x = dd.Variable((N, M), nonneg=True)
    resource_constraints = [x[i, :].sum() <= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= j for j in range(M)]
    
    # pull two elements
    objective = dd.Maximize(0 * x[0, 0] + 0 * x[1, 1] + 0 * x[2, 2])

    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=2, solver=dd.ECOS, rho=1, num_iter=5)
    print("DeDe:", result_dede)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve()
    print("CVXPY:", result_cvxpy)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01)
    print('=== Passed LP ADD zero test ===') 


def test_sum1():
    N, M = 10, 10
    x = dd.Variable((N, M), nonneg=True)
    resource_constraints = [x[i, :].sum() >= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= j for j in range(M)]
    objective = dd.Maximize(dd.sum(x))

    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=2, solver=dd.ECOS, rho=1, num_iter=5)
    print("DeDe:", result_dede)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve()
    print("CVXPY:", result_cvxpy)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01)
    print('=== Passed LP SUM test #1 ===')


def test_sum2():
    N, M = 10, 10
    x = dd.Variable((N, M), nonneg=True)
    resource_constraints = [x[i, :].sum() <= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= 1 for j in range(M)]
    objective = dd.Maximize(dd.sum(x))

    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=2, solver=dd.ECOS, rho=10, num_iter=7)
    print("DeDe:", result_dede)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve()
    print("CVXPY:", result_cvxpy)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01)
    print('=== Passed LP SUM test #2 ===')


def test_multiply1():
    N, M = 10, 10
    x = dd.Variable((N, M), nonneg=True)
    resource_constraints = [x[i, :].sum() >= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= j for j in range(M)]
    w = np.empty((N, M))
    for i in range(N):
        for j in range(M):
            w[i][j] = i - j
    objective = dd.Maximize(dd.sum(dd.multiply(x, w)))

    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=2, solver=dd.ECOS, rho=10, num_iter=15)
    print("DeDe:", result_dede)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve()
    print("CVXPY:", result_cvxpy)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01, abs_tol=0.1)
    print('=== Passed LP MULTIPLY test #1 ===')


def test_multiply2():
    N, M = 10, 10
    x = dd.Variable((N, M), nonneg=True)
    resource_constraints = [x[i, :].sum() >= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= j for j in range(M)]
    w = np.empty((N, M))
    for i in range(N):
        for j in range(M):
            w[i][j] = i * j / (i + j + 1)
    objective = dd.Maximize(dd.sum(dd.multiply(x, w)))

    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=2, solver=dd.ECOS, rho=1, num_iter=20)
    print("DeDe:", result_dede)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve()
    print("CVXPY:", result_cvxpy)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01, abs_tol=0.1)
    print('=== Passed LP MULTIPLY test #2 ===')


def test_multiply_zero():
    N, M = 3, 4
    x = dd.Variable((N, M), nonneg=True)
    resource_constraints = [x[i, :].sum() >= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= j for j in range(M)]
    w = np.zeros((N, M))
    w[0][2] = 1
    objective = dd.Maximize(dd.sum(dd.multiply(x, w)))

    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=2, solver=dd.ECOS, rho=1, num_iter=10)
    print("DeDe:", result_dede)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve()
    print("CVXPY:", result_cvxpy)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01, abs_tol=0.1)
    print('=== Passed LP MULTIPLY zero test ===')


if __name__ == '__main__':
    test_add1()
    test_add2()
    test_add_zero()

    test_sum1()
    test_sum2()

    test_multiply1()
    test_multiply2()
    test_multiply_zero()