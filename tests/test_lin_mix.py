#!/usr/bin/env python3

import dede as dd
import cvxpy as cp
import numpy as np
import math
from conftest import GUROBI_OPTS


def test_add1():
    N, M = 5, 5
    N1, N2 = N // 2, (N + 1) // 2

    x1 = dd.Variable((N1, M), integer=True)
    x2 = dd.Variable((N2, M), nonneg=True)

    resource_constraints = [x1[i, :] >= 0 for i in range(N1)] + \
                           [x1[i, :].sum() >= 2 * i for i in range(N1)] + \
                           [x2[i, :].sum() >= 2 * i for i in range(N2)]
    demand_constraints = [x1[:, j].sum() + x2[:, j].sum() <= 2 * 1.23456789 * j for j in range(M)]
    expr = 0
    for i in range(min(N, M)):
        if i < N1 and i % 2 == 0:
            expr += x1[i, i]
        elif i < N1 and i % 2 == 1:
            expr -= x1[i, i]
        elif i >= N1 and i % 2 == 0:
            expr += x2[i - N1, i]
        elif i >= N1 and i % 2 == 1:
            expr -= x2[i - N1, i]
    objective = dd.Maximize(expr)

    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=1, solver=dd.GUROBI, rho=1, num_iter=20, **GUROBI_OPTS)
    print("DeDe:", result_dede)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve()
    print("CVXPY:", result_cvxpy)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01)
    print('=== Passed MILP ADD test #1 ===')


def test_add2():
    N, M = 5, 5
    N1, N2 = N // 2, (N + 1) // 2

    x1 = dd.Variable((N1, M), nonneg=True)
    x2 = dd.Variable((N2, M), integer=True)

    resource_constraints = [x2[i, :] >= 0 for i in range(N2)] + \
                           [x1[i, :].sum() <= 1.5 * i for i in range(N1)] + \
                           [x2[i, :].sum() <= 1.5 * i for i in range(N2)]
    demand_constraints = [x1[:, j].sum() + x2[:, j].sum() <= j for j in range(M)]
    objective = dd.Maximize(x1[0, 0] + x1[0, M - 1] + x2[N2 - 1, 0] + x2[N2 - 1, M - 1])

    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=1, solver=dd.GUROBI, rho=0.5, num_iter=15, **GUROBI_OPTS)
    print("DeDe:", result_dede)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve()
    print("CVXPY:", result_cvxpy)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01)
    print('=== Passed MILP ADD test #2 ===')


def test_sum1():
    N, M = 5, 5
    N1, N2 = N // 2, (N + 1) // 2

    x1 = dd.Variable((N1, M), integer=True)
    x2 = dd.Variable((N2, M), nonneg=True)

    resource_constraints = [x1[i, :] >= 0 for i in range(N1)] + \
                           [x1[i, :].sum() >= i for i in range(N1)] + \
                           [x2[i, :].sum() >= i for i in range(N2)]
    demand_constraints = [x1[:, j].sum() + x2[:, j].sum() <= 2.3456 * j for j in range(M)]
    objective = dd.Maximize(dd.sum(x1) + dd.sum(x2))

    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=1, solver=dd.GUROBI, rho=1, num_iter=15, **GUROBI_OPTS)
    print("DeDe:", result_dede)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve()
    print("CVXPY:", result_cvxpy)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01)
    print('=== Passed MILP SUM test #1 ===')


def test_sum2():
    N, M = 5, 5
    N1, N2 = N // 2, (N + 1) // 2

    x1 = dd.Variable((N1, M), nonneg=True)
    x2 = dd.Variable((N2, M), integer=True)

    resource_constraints = [x1[i, :].sum() <= i for i in range(N1)] + \
                           [x2[i, :].sum() <= i for i in range(N2)]
    demand_constraints = [x1[:, j].sum() + x2[:, j].sum() <= 1 for j in range(M)]
    objective = dd.Maximize(dd.sum(x1) + dd.sum(x2))

    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=1, solver=dd.GUROBI, rho=1, num_iter=15, **GUROBI_OPTS)
    print("DeDe:", result_dede)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve()
    print("CVXPY:", result_cvxpy)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01)
    print('=== Passed MILP SUM test #2 ===')


def test_multiply1():
    N, M = 5, 5
    N1, N2 = N // 2, (N + 1) // 2

    x1 = dd.Variable((N1, M), integer=True)
    x2 = dd.Variable((N2, M), nonneg=True)

    resource_constraints = [x1[i, :] >= 0 for i in range(N1)] + \
                           [x1[i, :].sum() >= i for i in range(N1)] + \
                           [x2[i, :].sum() >= i for i in range(N2)]
    demand_constraints = [x1[:, j].sum() + x2[:, j].sum() <= j for j in range(M)]
    w = np.empty((N, M))
    for i in range(N):
        for j in range(M):
            w[i][j] = i - j
    objective = dd.Maximize(dd.sum(dd.multiply(x1, w[:N1])) + dd.sum(dd.multiply(x2, w[N1:])))

    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=1, solver=dd.GUROBI, rho=3, num_iter=25, **GUROBI_OPTS)
    print("DeDe:", result_dede)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve()
    print("CVXPY:", result_cvxpy)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01)
    print('=== Passed MILP MULTIPLY test #1 ===')


def test_multiply2():
    N, M = 5, 5
    N1, N2 = N // 2, (N + 1) // 2

    x1 = dd.Variable((N1, M), nonneg=True)
    x2 = dd.Variable((N2, M), integer=True)

    resource_constraints = [x2[i, :] >= 0 for i in range(N1)] + \
                           [x1[i, :].sum() >= i for i in range(N1)] + \
                           [x2[i, :].sum() >= i for i in range(N2)]
    demand_constraints = [x1[:, j].sum() + x2[:, j].sum() <= j for j in range(M)]
    w = np.empty((N, M))
    for i in range(N):
        for j in range(M):
            w[i][j] = i * j / (i + j + 1)
    objective = dd.Maximize(dd.sum(dd.multiply(x1, w[:N1])) + dd.sum(dd.multiply(x2, w[N1:])))

    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=1, solver=dd.GUROBI, rho=0.1, num_iter=31, **GUROBI_OPTS)
    print("DeDe:", result_dede)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve()
    print("CVXPY:", result_cvxpy)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01, abs_tol=0.1)
    print('=== Passed MILP MULTIPLY test #2 ===')


def test_boolean():
    M = 5 
    x1 = dd.Variable(M, boolean=True)
    x2 = dd.Variable(M, nonneg=True)
    x3 = dd.Variable(M, integer=True)

    resource_constraints = [x2.sum() <= 12.3456] + [x3.sum() <= 10]
    demand_constraints = [x1[j] + x2[j] + x3[j] >= 4 for j in range(M)]
    objective = dd.Maximize(dd.sum(x1) + dd.sum(x2) + dd.sum(x3))
    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=1, solver=dd.GUROBI, rho=1, num_iter=15, **GUROBI_OPTS)
    print("DeDe:", result_dede)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve(solver=cp.ECOS_BB)
    print("CVXPY:", result_cvxpy)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01)
    print('=== Passed MIXED BOOLEAN test ===')


if __name__ == '__main__':
    test_add1()
    test_add2()

    test_sum1()
    test_sum2()

    test_multiply1()
    test_multiply2()

    test_boolean()