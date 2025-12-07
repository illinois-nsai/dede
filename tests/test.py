#!/usr/bin/env python3

import dede as dd
import cvxpy as cp
import numpy as np
from scipy.optimize import minimize
import math
import time


def test(n):
    N, M = n, n
    x = dd.Variable((N, M), nonneg=True)
    resource_constraints = [x[i, :].sum() >= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= j for j in range(M)]
    
    objective = dd.Maximize(dd.sum(x))

    prob = dd.Problem(objective, resource_constraints, demand_constraints)
    start = time.time()
    result_dede = prob.solve(num_cpus=10, solver=dd.ECOS)
    time_dede = time.time() - start

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    start = time.time()
    result_cvxpy = cvxpy_prob.solve()
    time_cvxpy = time.time() - start

    with open("timing.txt", "a") as f:
        f.write(f"{n} {time_dede} {time_cvxpy} {result_dede} {result_cvxpy}\n")


if __name__ == '__main__':
    for i in range(500, 10001, 500):
        test(i)