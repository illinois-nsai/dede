#!/usr/bin/env python3

import dede as dd
import cvxpy as cp
import numpy as np
import math


def add1():
    N, M = 5, 5
    x = dd.Variable((N, M), integer=True)
    resource_constraints = [x >= 0] + [x[i, :].sum() >= 2 * i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= 2 * 1.23456789 * j for j in range(M)]
    expr = 0
    for i in range(min(N, M)):
        if i % 2 == 0:
            expr += x[i, i]
        else:
            expr -= x[i, i]
    objective = dd.Maximize(expr)

    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=4, solver=dd.ECOS_BB, rho=0.1, num_iter=50)
    print("DeDe:", result_dede)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve()
    print("CVXPY:", result_cvxpy)
    print(x.value)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01)
    print('=== Passed ILP ADD test #1 ===')


def add2():
    N, M = 5, 5
    x = dd.Variable((N, M), integer=True)
    resource_constraints = [x >= 0] + [x[i, :].sum() <= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= j for j in range(M)]
    
    # pull two elements
    objective = dd.Maximize(x[N-1, M-1] + x[N//2, M//2])

    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=4, solver=dd.ECOS_BB, rho=0.1, num_iter=40)
    print("DeDe:", result_dede)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve()
    print("CVXPY:", result_cvxpy)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01)
    print('=== Passed ILP ADD test #2 ===')


def constant1():
    N, M = 5, 5
    x = dd.Variable((N, M), integer=True)
    resource_constraints = [x >= 0] + [x[i, :].sum() <= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= j for j in range(M)]
    
    objective = dd.Maximize(2)

    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=4, solver=dd.ECOS_BB, rho=1, num_iter=5)
    print("DeDe:", result_dede)

    assert math.isclose(result_dede, 2, rel_tol=0.01)
    print('=== Passed ILP CONSTANT test #1 ===') 


def constant2():
    N, M = 5, 5
    x = dd.Variable((N, M), integer=True)
    resource_constraints = [x >= 0] + [x[i, :].sum() <= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= j for j in range(M)]
    
    objective = dd.Maximize(x[N-1, M-1] + 2)

    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=4, solver=dd.ECOS_BB, rho=0.1, num_iter=15)
    print("DeDe:", result_dede)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve()
    print("CVXPY:", result_cvxpy)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01)
    print('=== Passed ILP CONSTANT test #2 ===')


def quadratic():
    N, M = 10, 10
    x = dd.Variable((N, M), integer=True)

    resource_constraints = [x[i, :].sum() >= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= j for j in range(M)]
    objective = dd.Minimize(dd.sum_squares(x))
    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=4, solver=dd.ECOS_BB, rho=0.1, num_iter=15)
    print("DeDe:", result_dede)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve(solver=cp.ECOS_BB)
    print("CVXPY:", result_cvxpy)
    print(x.value)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01)
    print('=== Passed ILP QUADRATIC test ===')


if __name__ == '__main__':
    add1()
    add2()

    constant1()
    constant2()

    quadratic()