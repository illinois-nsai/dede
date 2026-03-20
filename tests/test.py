#!/usr/bin/env python3

import math

import cvxpy as cp
import numpy as np
from scipy.optimize import linprog

import dede as dd


def test_hard():
    N, M = 125, 125
    x = dd.Variable((N, M), nonneg=True)

    w = np.loadtxt("weights.txt")    
    w = w.reshape((N, M))
    bn = np.loadtxt("bn.txt")
    bm = np.loadtxt("bm.txt")

    resource_constraints = [x[i, :].sum() >= bn[i] for i in range(N)]
    demand_constraints = [x[:, j].sum() >= bm[j] for j in range(M)]
    objective = dd.Minimize(dd.sum(dd.multiply(x, w)))

    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=10, solver=dd.ECOS, rho=3.5, num_iter=None)
    print("DeDe:", result_dede)

    '''
    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve()
    print("CVXPY:", result_cvxpy)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01, abs_tol=0.1)
    print("=== Passed LP MULTIPLY test #1 ===")
    '''


def test_scipy():

    N, M = 125, 125

    w = np.loadtxt("weights.txt").reshape(N, M)
    bn = np.loadtxt("bn.txt")
    bm = np.loadtxt("bm.txt")

    # flatten variables
    c = w.flatten()

    # --- row constraints ---
    A_rows = np.zeros((N, N*M))
    for i in range(N):
        for j in range(M):
            A_rows[i, i*M + j] = -1
    b_rows = -bn

    # --- column constraints ---
    A_cols = np.zeros((M, N*M))
    for j in range(M):
        for i in range(N):
            A_cols[j, i*M + j] = -1
    b_cols = -bm

    A_ub = np.vstack([A_rows, A_cols])
    b_ub = np.concatenate([b_rows, b_cols])

    bounds = [(0, None)] * (N*M)

    res = linprog(
        c,
        A_ub=A_ub,
        b_ub=b_ub,
        bounds=bounds,
        method="highs"
    )

    print("SciPy objective:", res.fun)

    x = res.x.reshape(N, M)
    return x


if __name__ == "__main__":
    #test_scipy()
    test_hard()
