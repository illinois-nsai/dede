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
    w = 9 * np.random.uniform(0, 1, (N, M)) + 1
    bn = 9 * np.random.uniform(0, 1, (N,)) + 1
    bm = 9 * np.random.uniform(0, 1, (M,)) + 1

    resource_constraints = [x[i, :].sum() >= bn[i] for i in range(N)]
    demand_constraints = [x[:, j].sum() >= bm[j] for j in range(M)]
    
    objective = dd.Minimize(dd.sum(dd.multiply(x, w)))

    prob = dd.Problem(objective, resource_constraints, demand_constraints)
    start = time.time()
    result_dede = prob.solve(num_cpus=10, solver=dd.ECOS)
    time_dede = time.time() - start

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    start = time.time()
    result_cvxpy = cvxpy_prob.solve()
    time_cvxpy = time.time() - start

    with open("timing.txt", "a") as f:
        f.write(f"{n} {time_dede} {time_cvxpy}\n")


if __name__ == '__main__':
    for i in range(2125, 10000, 500):
        test(i)
