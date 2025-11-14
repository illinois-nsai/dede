#!/usr/bin/env python3

import dede as dd
import cvxpy as cp
import numpy as np
from scipy.optimize import minimize
import math


def test():
    N, M = 1000, 1000
    x = dd.Variable((N, M), nonneg=True)
    resource_constraints = [x[i, :].sum() >= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= j for j in range(M)]

    w = np.empty((N, M))
    for i in range(N):
        for j in range(M):
            w[i][j] = (i + 1) ** 3 / (j + 1)
    
    objective = dd.Maximize(dd.sum(dd.multiply(w, x)))

    #prob = dd.Problem(objective, resource_constraints, demand_constraints)
    print("done compiling")

    result_dede = 0
    #result_dede = prob.solve(num_cpus=10, solver=dd.ECOS)
    print("DeDe:", result_dede)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve()
    print("CVXPY:", result_cvxpy)
    print("CVXPY solve time:", cvxpy_prob.solver_stats.solve_time)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01)
    print('=== Passed LP ADD test #1 ===')


if __name__ == '__main__':
    test()