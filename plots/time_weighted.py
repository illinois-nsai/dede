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
    value_dede = x.value
    time_dede = time.time() - start

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    start = time.time()
    result_cvxpy = cvxpy_prob.solve()
    value_cvxpy = x.value
    time_cvxpy = time.time() - start

    with open("timing.txt", "a") as f:
        f.write(f"{n} {time_dede} {time_cvxpy} {result_dede} {result_cvxpy}\n")

        for x1, x2 in zip(value_cvxpy.flatten(), value_dede.flatten()):
            if x1 > 1e-5 or x2 > 1e-5:
                f.write(f"{x1:.4f} {x2:.4f}\n")
    
    print("NORM IS:", np.linalg.norm(value_cvxpy.flatten() - value_dede.flatten(), 2))

    with open("weights.txt", "a") as f:
        for x1 in w.flatten():
            f.write(f"{x1}\n")
    with open("bn.txt", "a") as f:
        for x1 in bn:
            f.write(f"{x1}\n")
    with open("bm.txt", "a") as f:
        for x1 in bm:
            f.write(f"{x1}\n")


if __name__ == '__main__':
    test(125)
