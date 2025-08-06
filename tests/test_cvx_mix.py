#!/usr/bin/env python3

import dede as dd
import cvxpy as cp
import numpy as np
import math


def test_quadratic():
    N, M = 5, 5
    N1, N2 = N // 2, (N + 1) // 2

    x1 = dd.Variable((N1, M), integer=True)
    x2 = dd.Variable((N2, M), nonneg=True)

    resource_constraints = [x1[i, :].sum() >= 1.2 * i for i in range(N1)] + \
                           [x2[i, :].sum() >= 3.5 * i for i in range(N2)]
    demand_constraints = [x1[:, j].sum() + x2[:, j].sum() >= 2.23 * j for j in range(M)]
    objective = dd.Minimize(dd.quad_over_lin(x1, 1) + dd.quad_over_lin(x2, 1))
    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=2, solver=dd.GUROBI, rho=1, num_iter=15)
    print("DeDe:", result_dede)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve(solver=cp.ECOS_BB)
    print("CVXPY:", result_cvxpy)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01)
    print('=== Passed MIXED QUADRATIC test ===')


def test_quadratic_weighted():
    N, M = 5, 5
    N1, N2 = N // 2, (N + 1) // 2

    x1 = dd.Variable((N1, M), nonneg=True)
    x2 = dd.Variable((N2, M), integer=True)

    resource_constraints = [x1[i, :].sum() >= 9.6 * i for i in range(N1)] + \
                           [x2[i, :].sum() >= i for i in range(N2)]
    demand_constraints = [x1[:, j].sum() + x2[:, j].sum() >= 5.5 * j for j in range(M)]
    w = np.empty((N, M))
    for i in range(N):
        for j in range(M):
            w[i][j] = np.sqrt((i + 1) * (j + 1))
    objective = dd.Minimize(dd.quad_over_lin(dd.multiply(x1, w[:N1]), 1) + \
                            dd.quad_over_lin(dd.multiply(x2, w[N1:]), 1))

    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=2, solver=dd.GUROBI, rho=10, num_iter=20)
    print("DeDe:", result_dede)
    
    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve(solver=cp.ECOS_BB)
    print("CVXPY:", result_cvxpy)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01)
    print('=== Passed MIXED QUADRATIC weighted test ===')


def test_boolean_quadratic():
    M = 5 
    x1 = dd.Variable(M, boolean=True)
    x2 = dd.Variable(M, nonneg=True)
    x3 = dd.Variable(M, integer=True)

    resource_constraints = [x2.sum() >= 12.3456] + [x3.sum() >= 10]
    demand_constraints = [x1[j] + x2[j] + x3[j] >= 4 * j for j in range(M)]
    objective = dd.Minimize(dd.quad_over_lin(x1, 1) + \
                            dd.quad_over_lin(x2, 1) + \
                            dd.quad_over_lin(x3, 1))
    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=2, solver=dd.GUROBI, rho=5, num_iter=15)
    print("DeDe:", result_dede)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve(solver=cp.ECOS_BB)
    print("CVXPY:", result_cvxpy)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01)
    print('=== Passed MIXED BOOL QUADRATIC test ===') 


def test_boolean_quadratic_weighted():
    M = 5 
    x1 = dd.Variable(M, boolean=True)
    x2 = dd.Variable(M, nonneg=True)
    x3 = dd.Variable(M, integer=True)

    resource_constraints = [x2.sum() >= 12.3456] + [x3.sum() >= 10]
    demand_constraints = [x1[j] + x2[j] + x3[j] >= 4 * j for j in range(M)]

    w1 = np.arange(1, M+1)
    w2 = np.empty(M)
    w3 = np.empty(M)
    for i in range(M):
        w2[i] = np.sin(i ** 2)
        w3[i] = np.cos(i ** 3)

    objective = dd.Minimize(dd.quad_over_lin(dd.multiply(x1, w1), 1) + \
                            dd.quad_over_lin(dd.multiply(x2, w2), 1) + \
                            dd.quad_over_lin(dd.multiply(x3, w3), 1))
    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=2, solver=dd.GUROBI, rho=0.5, num_iter=15)
    print("DeDe:", result_dede)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve(solver=cp.ECOS_BB)
    print("CVXPY:", result_cvxpy)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01)
    print('=== Passed MIXED BOOL QUADRATIC weighted test ===')


if __name__ == '__main__':
    test_quadratic()
    test_quadratic_weighted()

    test_boolean_quadratic()
    test_boolean_quadratic_weighted()