#!/usr/bin/env python3

import dede as dd
import cvxpy as cp
import numpy as np
import math
from conftest import GUROBI_OPTS


def test_quadratic():
    N, M = 10, 10
    x = dd.Variable((N, M), integer=True)

    resource_constraints = [x[i, :].sum() >= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= j for j in range(M)]
    objective = dd.Minimize(dd.quad_over_lin(x, 1))
    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=2, solver=dd.GUROBI, rho=1, num_iter=25, **GUROBI_OPTS)
    print("DeDe:", result_dede)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve(solver=cp.ECOS_BB)
    print("CVXPY:", result_cvxpy)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01)
    print('=== Passed INTEGER QUADRATIC test ===')


def test_quadratic_weighted():
    N, M = 5, 5
    x = dd.Variable((N, M), integer=True)

    resource_constraints = [x[i, :].sum() >= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= j for j in range(M)]
    w = np.empty((N, M))
    for i in range(N):
        for j in range(M):
            w[i][j] = (i + 1) * (j + 1)
    objective = dd.Minimize(dd.quad_over_lin(dd.multiply(x, w), 1))
    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=2, solver=dd.GUROBI, rho=50, num_iter=20, **GUROBI_OPTS)
    print("DeDe:", result_dede)
    
    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve(solver=cp.ECOS_BB)
    print("CVXPY:", result_cvxpy)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01)
    print('=== Passed INTEGER QUADRATIC weighted test ===')


def test_boolean_quadratic():
    N, M = 5, 5
    x = dd.Variable((N, M), boolean=True)

    resource_constraints = [x[i, :].sum() >= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= j for j in range(M)]
    objective = dd.Minimize(dd.quad_over_lin(x, 3))
    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=2, solver=dd.GUROBI, rho=10, num_iter=15, **GUROBI_OPTS)
    print("DeDe:", result_dede)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve(solver=cp.ECOS_BB)
    print("CVXPY:", result_cvxpy)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01)
    print('=== Passed BOOLEAN QUADRATIC test ===')


if __name__ == '__main__':
    test_quadratic()
    test_quadratic_weighted()

    test_boolean_quadratic()