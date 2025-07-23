#!/usr/bin/env python3

import dede as dd
import cvxpy as cp
import numpy as np
import math


def sum():
    N, M = 10, 10
    x = dd.Variable((N, M), nonneg=True)
    y = dd.Variable((N, M), boolean=True)
    resource_constraints = [x[i, :].sum() >= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= 2 * j for j in range(M)]
    link_constraints = [x[i, j] <= 10 * y[i, j] for i in range(N) for j in range(M)]
    objective = dd.Maximize(dd.sum(x) - dd.sum([dd.sum(y[:, j]) for j in range(M)]))

    prob = dd.Problem(objective, resource_constraints, demand_constraints + link_constraints)

    result_dede = prob.solve(num_cpus=4, solver=dd.GUROBI, rho=0.2, num_iter=100)
    print("DeDe:", result_dede)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints + link_constraints)
    result_cvxpy = cvxpy_prob.solve()
    print("CVXPY:", result_cvxpy)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01)
    print("=== Passed MILP SUM test ===")


def quadratic():
    N, M = 10, 10
    x = dd.Variable((N, M), integer=True)
    y = dd.Variable((N, M), nonneg=True)
    resource_constraints = [x[i, :] >= 0 for i in range(N)] + \
                           [x[i, :].sum() + y[i, :].sum() >= i for i in range(N)]
    demand_constraints = [x[:, j].sum() + y[:, j].sum() <= j for j in range(M)]
    objective = dd.Minimize(dd.sum_squares(x - y))

    prob = dd.Problem(objective, resource_constraints, demand_constraints)
    results = []
    result_dede = 0
    for i in range(1, 10):
        result_dede = prob.solve(num_cpus=4, solver=dd.GUROBI, rho=0.1*i, num_iter=100)
        results.append(result_dede)
    for i in range(1, 10):
        print("DeDe", i*0.1, results[i-1])

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve(solver=cp.ECOS_BB)
    print("CVXPY:", result_cvxpy)
    print(x.value)
    print(y.value)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01)
    print("=== Passed MILP QUADRATIC test ===")


if __name__ == '__main__':
    #sum()
    quadratic()