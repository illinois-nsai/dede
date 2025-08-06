#!/usr/bin/env python3

import dede as dd
from dede.constraints_utils import func
import cvxpy as cp
import numpy as np
from scipy.optimize import minimize
import math
import time




def test_sum1():
    N, M = 10, 10
    x = dd.Variable((N, M), integer=True)
    resource_constraints = [x[i, :] >= 0 for i in range(N)] + [x[i, :].sum() >= 2 * i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= 2 * 1.23456789 * j for j in range(M)]
    expr = 0
    for i in range(min(N, M)):
        if i % 2 == 0:
            expr += x[i, i]
        else:
            expr -= x[i, i]
    objective = dd.Maximize(expr)


    prob = dd.Problem(objective, resource_constraints, demand_constraints)
    startd = time.time()
    result_dede = prob.solve(num_cpus=4, solver=dd.GUROBI, rho=0.5, num_iter=20)
    print("DTIME:", time.time() - startd)
    print("DeDe:", result_dede)


    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    startc = time.time()
    result_cvxpy = cvxpy_prob.solve()
    print("CTIME:", time.time() - startc)
    print("CVXPY:", result_cvxpy)


    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01)
    print('=== Passed LP SUM test #1 ===')


def test_sum2():
    N, M = 10, 10
    x = dd.Variable((N, M), integer=True)
    resource_constraints = func([x[i, :] >= 0 for i in range(N)] + [x[i, :].sum() >= 2 * i for i in range(N)])
    demand_constraints = func([x[:, j].sum() <= 2 * 1.23456789 * j for j in range(M)])
    expr = 0
    for i in range(min(N, M)):
        if i % 2 == 0:
            expr += x[i, i]
        else:
            expr -= x[i, i]
    objective = dd.Maximize(expr)


    prob = dd.Problem(objective, resource_constraints, demand_constraints)
    startd = time.time()
    result_dede = prob.solve(num_cpus=4, solver=dd.GUROBI, rho=0.5, num_iter=23)
    print("DTIME:", time.time() - startd)
    print("DeDe:", result_dede)


    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    startc = time.time()
    result_cvxpy = cvxpy_prob.solve()
    print("CTIME:", time.time() - startc)
    print("CVXPY:", result_cvxpy)


    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01)
    print('=== Passed LP SUM test #2 ===')


if __name__ == '__main__':
    sum1()
    sum2()

