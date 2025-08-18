#!/usr/bin/env python3

import dede as dd
import cvxpy as cp
import numpy as np
import math


def test_lin_cont():
    N, M = 10, 10
    x = dd.Variable((N, M), nonneg=True)
    resource_constraints = [x[i, :].sum() <= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= j for j in range(M)]
    w = np.empty((N, M))
    for i in range(N):
        for j in range(M):
            w[i][j] = i * j
    objective = dd.Maximize(dd.sum(dd.multiply(x, w)))

    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=2, solver=dd.GUROBI, rho=100, num_iter=1000)
    print("DeDe:", result_dede)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve()
    print("CVXPY:", result_cvxpy)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01)
    print('=== Passed CONTINUOUS LINEAR value test ===')


def test():
    N, M = 10, 10
    x = cp.Variable((N, M), nonneg=True)
    resource_constraints = [x[i, :].sum() >= i for i in range(N)]
    demand_constraints = [x[:, j].sum() >= j for j in range(M)]
    #objective = cp.Minimize(cp.norm(x, "fro") ** 2)
    objective = cp.Minimize(cp.sum_squares(x))

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve(solver=cp.ECOS_BB)
    print(result_cvxpy)
    print(x.value)


if __name__ == '__main__':
    np.set_printoptions(precision=4, suppress=True)
    #test()
    test_lin_cont()