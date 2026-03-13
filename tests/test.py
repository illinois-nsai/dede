#!/usr/bin/env python3

import math

import cvxpy as cp
import numpy as np

import dede as dd


def test_hard():
    N, M = 125, 125
    x = dd.Variable((N, M), nonneg=True)

    w = np.loadtxt("weights.txt")    
    w = w.reshape((N, M))
    print(w)
    bn = np.loadtxt("bn.txt")
    print(bn)
    bm = np.loadtxt("bm.txt")

    resource_constraints = [x[i, :].sum() >= bn[i] for i in range(N)]
    demand_constraints = [x[:, j].sum() >= bm[j] for j in range(M)]
    objective = dd.Minimize(dd.sum(dd.multiply(x, w)))

    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=10, solver=dd.ECOS, rho=5, num_iter=100)
    print("DeDe:", result_dede)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve()
    print("CVXPY:", result_cvxpy)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01, abs_tol=0.1)
    print("=== Passed LP MULTIPLY test #1 ===")


if __name__ == "__main__":
    test_hard()
