#!/usr/bin/env python3

import dede as dd
import cvxpy as cp
import numpy as np
import math


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

    result_dede = prob.solve(num_cpus=4, solver=dd.GUROBI, rho=1, num_iter=20)
    print("DeDe:", result_dede)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve()
    print("CVXPY:", result_cvxpy)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01)
    print('=== Passed ILP ADD test #1 ===')


def test_add2():
    N, M = 10, 10
    x = dd.Variable((N, M), integer=True)
    resource_constraints = [x >= 0] + [x[i, :].sum() <= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= j for j in range(M)]
    
    # pull two elements
    objective = dd.Maximize(x[N-1, M-1] + x[N//2, M//2])

    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=4, solver=dd.GUROBI, rho=0.1, num_iter=30)
    print("DeDe:", result_dede)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve()
    print("CVXPY:", result_cvxpy)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01)
    print('=== Passed ILP ADD test #2 ===')


def test_constant1():
    N, M = 5, 5
    x = dd.Variable((N, M), integer=True)
    resource_constraints = [x[i, :] >= 0 for i in range(N)] + [x[i, :].sum() <= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= j for j in range(M)]
    
    objective = dd.Maximize(2)

    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=4, solver=dd.GUROBI, rho=1, num_iter=5)
    print("DeDe:", result_dede)

    assert math.isclose(result_dede, 2, rel_tol=0.01)
    print('=== Passed ILP CONSTANT test #1 ===') 


def test_constant2():
    N, M = 10, 10
    x = dd.Variable((N, M), integer=True)
    resource_constraints = [x[i, :] >= 0 for i in range(N)] + [x[i, :].sum() <= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= j for j in range(M)]
    
    objective = dd.Maximize(x[N-1, M-1] + 2)

    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=4, solver=dd.GUROBI, rho=0.5, num_iter=20)
    print("DeDe:", result_dede)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve()
    print("CVXPY:", result_cvxpy)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01)
    print('=== Passed ILP CONSTANT test #2 ===')


def test_sum1():
    N, M = 5, 5
    x = dd.Variable((N, M), integer=True)
    resource_constraints = [x[i, :].sum() >= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= j for j in range(M)]
    objective = dd.Maximize(dd.sum(x))

    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=4, solver=dd.GUROBI, rho=1, num_iter=30)
    print("DeDe:", result_dede)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve()
    print("CVXPY:", result_cvxpy)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01)
    print('=== Passed ILP SUM test #1 ===')


def test_sum2():
    N, M = 5, 5
    x = dd.Variable((N, M), integer=True)
    resource_constraints = [x[i, :].sum() <= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= 1 for j in range(M)]
    objective = dd.Maximize(dd.sum(x))

    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=4, solver=dd.GUROBI, rho=1, num_iter=7)
    print("DeDe:", result_dede)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve()
    print("CVXPY:", result_cvxpy)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01)
    print('=== Passed ILP SUM test #2 ===')


def test_multiply1():
    N, M = 10, 10
    x = dd.Variable((N, M), integer=True)
    resource_constraints = [x[i, :] >= 0 for i in range(N)] + [x[i, :].sum() >= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= j for j in range(M)]
    w = np.empty((N, M))
    for i in range(N):
        for j in range(M):
            w[i][j] = i - j
    objective = dd.Maximize(dd.sum(dd.multiply(x, w)))

    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=4, solver=dd.GUROBI, rho=1, num_iter=50)
    print("DeDe:", result_dede)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve()
    print("CVXPY:", result_cvxpy)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01, abs_tol=0.1)
    print('=== Passed ILP MULTIPLY test #1 ===')


def test_multiply2():
    N, M = 10, 10
    x = dd.Variable((N, M), integer=True)
    resource_constraints = [x[i, :] >= 0 for i in range(N)] + [x[i, :].sum() >= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= j for j in range(M)]
    w = np.empty((N, M))
    for i in range(N):
        for j in range(M):
            w[i][j] = i * j / (i + j + 1)
    objective = dd.Maximize(dd.sum(dd.multiply(x, w)))

    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=4, solver=dd.GUROBI, rho=0.4755, num_iter=90)
    print("DeDe:", result_dede)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve()
    print("CVXPY:", result_cvxpy)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01, abs_tol=0.1)
    print('=== Passed ILP MULTIPLY test #2 ===')


def test_boolean():
    N, M = 10, 10
    x = dd.Variable((N, M), boolean=True)

    resource_constraints = [x[i, :].sum() >= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= j for j in range(M)]
    objective = dd.Maximize(dd.sum(x))
    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=4, solver=dd.GUROBI, rho=10, num_iter=15)
    print("DeDe:", result_dede)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve(solver=cp.ECOS_BB)
    print("CVXPY:", result_cvxpy)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01)
    print('=== Passed BOOLEAN LP test ===')


if __name__ == '__main__':
    test_add1()
    '''
    test_add2()

    test_constant1()
    test_constant2()

    test_sum1()
    test_sum2()

    test_multiply1()
    test_multiply2()

    test_boolean()
    '''