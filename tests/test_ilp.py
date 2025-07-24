#!/usr/bin/env python3

import dede as dd
import cvxpy as cp
import numpy as np
import math


def add1():
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

    result_dede = prob.solve(num_cpus=4, solver=dd.GUROBI, rho=0.5, num_iter=23)
    print("DeDe:", result_dede)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve()
    print("CVXPY:", result_cvxpy)
    print(x.value)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01)
    print('=== Passed ILP ADD test #1 ===')


def add2():
    N, M = 10, 10
    x = dd.Variable((N, M), integer=True)
    resource_constraints = [x >= 0] + [x[i, :].sum() <= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= j for j in range(M)]
    
    # pull two elements
    objective = dd.Maximize(x[N-1, M-1] + x[N//2, M//2])

    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=4, solver=dd.GUROBI, rho=0.1, num_iter=30)
    print("DeDe:", result_dede)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve()
    print("CVXPY:", result_cvxpy)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01)
    print('=== Passed ILP ADD test #2 ===')


def constant1():
    N, M = 5, 5
    x = dd.Variable((N, M), integer=True)
    resource_constraints = [x[i, :] >= 0 for i in range(N)] + [x[i, :].sum() <= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= j for j in range(M)]
    
    objective = dd.Maximize(2)

    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=4, solver=dd.GUROBI, rho=1, num_iter=5)
    print("DeDe:", result_dede)

    assert math.isclose(result_dede, 2, rel_tol=0.01)
    print('=== Passed ILP CONSTANT test #1 ===') 


def constant2():
    N, M = 10, 10
    x = dd.Variable((N, M), integer=True)
    resource_constraints = [x[i, :] >= 0 for i in range(N)] + [x[i, :].sum() <= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= j for j in range(M)]
    
    objective = dd.Maximize(x[N-1, M-1] + 2)

    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=4, solver=dd.GUROBI, rho=0.5, num_iter=20)
    print("DeDe:", result_dede)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve()
    print("CVXPY:", result_cvxpy)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01)
    print('=== Passed ILP CONSTANT test #2 ===')


def sum1():
    N, M = 5, 5
    x = dd.Variable((N, M), integer=True)
    resource_constraints = [x[i, :].sum() >= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= j for j in range(M)]
    objective = dd.Maximize(dd.sum(x))

    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=4, solver=dd.GUROBI, rho=1, num_iter=30)
    print("DeDe:", result_dede)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve()
    print("CVXPY:", result_cvxpy)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01)
    print('=== Passed ILP SUM test #1 ===')


def sum2():
    N, M = 5, 5
    x = dd.Variable((N, M), integer=True)
    resource_constraints = [x[i, :].sum() <= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= 1 for j in range(M)]
    objective = dd.Maximize(dd.sum(x))

    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=4, solver=dd.GUROBI, rho=1, num_iter=7)
    print("DeDe:", result_dede)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve()
    print("CVXPY:", result_cvxpy)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01)
    print('=== Passed ILP SUM test #2 ===')


def multiply1():
    N, M = 10, 10
    x = dd.Variable((N, M), integer=True)
    resource_constraints = [x[i, :] >= 0 for i in range(N)] + [x[i, :].sum() >= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= j for j in range(M)]
    w = np.empty((N, M))
    for i in range(N):
        for j in range(M):
            w[i][j] = i - j
    objective = dd.Maximize(dd.sum(dd.multiply(x, w)))

    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=4, solver=dd.GUROBI, rho=1, num_iter=50)
    print("DeDe:", result_dede)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve()
    print("CVXPY:", result_cvxpy)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01, abs_tol=0.1)
    print('=== Passed ILP MULTIPLY test #1 ===')


def multiply2():
    N, M = 10, 10
    x = dd.Variable((N, M), integer=True)
    resource_constraints = [x[i, :] >= 0 for i in range(N)] + [x[i, :].sum() >= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= j for j in range(M)]
    w = np.empty((N, M))
    for i in range(N):
        for j in range(M):
            w[i][j] = i * j / (i + j + 1)
    objective = dd.Maximize(dd.sum(dd.multiply(x, w)))

    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=4, solver=dd.GUROBI, rho=0.5, num_iter=85)
    print("DeDe:", result_dede)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve()
    print("CVXPY:", result_cvxpy)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01, abs_tol=0.1)
    print('=== Passed ILP MULTIPLY test #2 ===')


def quadratic():
    N, M = 10, 10
    x = dd.Variable((N, M), integer=True)

    resource_constraints = [x[i, :].sum() >= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= j for j in range(M)]
    objective = dd.Minimize(dd.sum_squares(x))
    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=4, solver=dd.GUROBI, rho=0.5, num_iter=35)
    print("DeDe:", result_dede)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve(solver=cp.ECOS_BB)
    print("CVXPY:", result_cvxpy)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01)
    print('=== Passed ILP QUADRATIC test ===')


def boolean():
    N, M = 20, 20
    x = dd.Variable((N, M), boolean=True)

    resource_constraints = [x[i, :].sum() >= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= j for j in range(M)]
    objective = dd.Maximize(dd.sum(x))
    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=4, solver=dd.GUROBI, rho=10, num_iter=35)
    print("DeDe:", result_dede)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve(solver=cp.ECOS_BB)
    print("CVXPY:", result_cvxpy)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01)
    print('=== Passed BOOLEAN LP test ===')


if __name__ == '__main__':
    add1()
    add2()

    constant1()
    constant2()

    sum1()
    sum2()

    multiply1()
    multiply2()

    quadratic()

    boolean()