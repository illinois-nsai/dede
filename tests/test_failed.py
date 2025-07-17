#!/usr/bin/env python3

import dede as dd
import cvxpy as cp
import numpy as np
from scipy.optimize import minimize
import math


def log():
    N, M = 10, 10
    x = dd.Variable((N, M), nonneg=True)

    # nonnegative log values: x >= 1
    resource_constraints = [x >= 1] + [x[i, :].sum() <= (i + 1) * M for i in range(N)]
    demand_constraints = [x[:, j].sum() <= (j + 1) * N for j in range(M)]
    
    # write in separable form, dd.sum(x) fails
    expr = dd.sum([dd.sum(dd.log(x[i, :])) for i in range(N)])
    objective = dd.Maximize(expr)

    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=4, solver=dd.ECOS, rho=0.07, num_iter=30)
    print("DeDe:", result_dede)


def sum2():
    N, M = 5, 5
    x = dd.Variable((N, M), integer=True)
    resource_constraints = [x[i, :].sum() <= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= 1 for j in range(M)]
    objective = dd.Maximize(dd.sum(x))

    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=4, solver=dd.ECOS_BB, rho=0.1, num_iter=50)
    print("DeDe:", result_dede)


def log2():
    N, M = 10, 10
    x = dd.Variable((N, M), integer=True)

    # nonnegative log values: x >= 1
    resource_constraints = [x >= 1] + [x[i, :].sum() <= (i + 1) * M for i in range(N)]
    demand_constraints = [x[:, j].sum() <= (j + 1) * N for j in range(M)]
    
    # write in separable form, dd.sum(x) fails
    expr = dd.sum([dd.sum(dd.log(x[i, :])) for i in range(N)])
    objective = dd.Maximize(expr)

    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=4, solver=dd.ECOS_BB, rho=1, num_iter=30)
    print("DeDe:", result_dede)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve(solver=cp.ECOS_BB)
    print("CVXPY:", result_cvxpy)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01)
    print('=== Passed ILP LOG test ===') 


if __name__ == '__main__':
    #log()
    log2()