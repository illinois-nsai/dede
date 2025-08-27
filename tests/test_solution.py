#!/usr/bin/env python3

import dede as dd
import numpy as np
from conftest import GUROBI_OPTS
import math


def test_lin_cont():
    N, M = 10, 10
    x = dd.Variable((N, M), nonneg=True)
    resource_constraints = [x[i, :].sum() >= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= j for j in range(M)]
    w = np.empty((N, M))
    for i in range(N):
        for j in range(M):
            w[i][j] = i * j / (i + j + 1)
    objective = dd.Maximize(dd.sum(dd.multiply(x, w)))

    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=2, solver=dd.ECOS, rho=1, num_iter=20)
    print("Returned result:", result_dede)

    result_solution = np.sum(np.multiply(x.value, w))
    print("Solution result:", result_solution)

    assert math.isclose(result_dede, result_solution, rel_tol=0.01)
    print('=== Passed CONTINUOUS LINEAR value test ===')


def test_cvx_cont():
    N, M = 10, 10
    x = dd.Variable((N, M), nonneg=True)

    # nonnegative log values: x >= 1
    resource_constraints = [x[i, :] >= 1 for i in range(N)] + [x[i, :].sum() <= (i + 1) * M for i in range(N)]
    demand_constraints = [x[:, j].sum() <= (j + 1) * N for j in range(M)]

    w = np.empty((N, M))
    for i in range(N):
        for j in range(M):
            w[i][j] = i + j
    expr = dd.sum([dd.sum(dd.multiply(w[i, :], dd.log(x[i, :]))) for i in range(N)])
    objective = dd.Maximize(expr)

    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=2, solver=dd.ECOS, rho=1, num_iter=50)
    print("Returned result:", result_dede)

    result_solution = np.sum(np.multiply(w, np.log(x.value)))
    print("Solution result:", result_solution)

    assert math.isclose(result_dede, result_solution, rel_tol=0.01)
    print('=== Passed CONTINUOUS CONVEX value test ===')


def test_lin_int():
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

    result_dede = prob.solve(num_cpus=2, solver=dd.GUROBI, rho=0.1, num_iter=25, **GUROBI_OPTS)
    print("Returned result:", result_dede)

    result_solution = np.sum(np.multiply(x.value, w))
    print("Solution result:", result_solution)

    assert math.isclose(result_dede, result_solution, rel_tol=0.01)
    print('=== Passed INTEGER LINEAR value test ===')


def test_cvx_int():
    N, M = 5, 5
    x = dd.Variable((N, M), integer=True)

    resource_constraints = [x[i, :].sum() >= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= j for j in range(M)]
    w = np.empty((N, M))
    for i in range(N):
        for j in range(M):
            w[i][j] = (i + 1) * (j + 1)
    objective = dd.Minimize(dd.quad_over_lin(dd.multiply(x, w), 1))
    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=2, solver=dd.GUROBI, rho=50, num_iter=20, **GUROBI_OPTS)
    print("Returned result:", result_dede)
    
    result_solution = np.sum(np.multiply(x.value, w) ** 2)
    print("Solution result:", result_solution)

    assert math.isclose(result_dede, result_solution, rel_tol=0.01)
    print('=== Passed INTEGER CONVEX value test ===')


def test_lin_mix():
    M = 5 
    x1 = dd.Variable(M, boolean=True)
    x2 = dd.Variable(M, nonneg=True)
    x3 = dd.Variable(M, integer=True)

    resource_constraints = [x2.sum() <= 12.3456] + [x3.sum() <= 10]
    demand_constraints = [x1[j] + x2[j] + x3[j] >= 4 for j in range(M)]
    objective = dd.Maximize(dd.sum(x1) + dd.sum(x2) + dd.sum(x3))
    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=2, solver=dd.GUROBI, rho=1, num_iter=15, **GUROBI_OPTS)
    print("Returned result:", result_dede)

    result_solution = np.sum(x1.value) + np.sum(x2.value) + np.sum(x3.value)
    print("Solution result:", result_solution)

    assert math.isclose(result_dede, result_solution, rel_tol=0.01)
    print('=== Passed MIXED LINEAR value test ===') 


def test_cvx_mix():
    M = 5 
    x1 = dd.Variable(M, boolean=True)
    x2 = dd.Variable(M, nonneg=True)
    x3 = dd.Variable(M, integer=True)

    resource_constraints = [x2.sum() >= 12.3456] + [x3.sum() >= 10]
    demand_constraints = [x1[j] + x2[j] + x3[j] >= 4 * j for j in range(M)]

    w1 = np.arange(1, M+1)
    w2 = np.empty(M)
    w3 = np.empty(M)
    for i in range(M):
        w2[i] = np.sin(i ** 2)
        w3[i] = np.cos(i ** 3)

    objective = dd.Minimize(dd.quad_over_lin(dd.multiply(x1, w1), 1) + \
                            dd.quad_over_lin(dd.multiply(x2, w2), 1) + \
                            dd.quad_over_lin(dd.multiply(x3, w3), 1))
    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=2, solver=dd.GUROBI, rho=0.5, num_iter=15, **GUROBI_OPTS)
    print("Returned result:", result_dede)

    result_solution = np.sum(np.multiply(x1.value, w1) ** 2) + \
                      np.sum(np.multiply(x2.value, w2) ** 2) + \
                      np.sum(np.multiply(x3.value, w3) ** 2)
    print("Solution result:", result_solution)

    assert math.isclose(result_dede, result_solution, rel_tol=0.01)
    print('=== Passed MIXED CONVEX value test ===')


def test_large():
    N, M = 100, 100
    x1 = dd.Variable(M, boolean=True)
    x2 = dd.Variable((N, M), nonneg=True)

    resource_constraints = [x2[i, :].sum() <= 1 for i in range(N)]
    demand_constraints = [x1[j] + x2[:, j].sum() <= j for j in range(M)]
    objective = dd.Maximize(dd.sum(x1) + dd.sum(x2))

    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=4, solver=dd.GUROBI, rho=1, num_iter=15, **GUROBI_OPTS)
    print("Returned result:", result_dede)

    result_solution = np.sum(x1.value) + np.sum(x2.value)
    print("Solution result:", result_solution)

    assert math.isclose(result_dede, result_solution, rel_tol=0.01)
    print('=== Passed LARGE value test ===')


if __name__ == '__main__':
    test_lin_cont()
    test_cvx_cont()

    test_lin_int()
    test_cvx_int()

    test_lin_mix()
    test_cvx_mix()

    test_large()