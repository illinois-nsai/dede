#!/usr/bin/env python3

import cvxpy as cp
import numpy as np
from conftest import GUROBI_OPTS, check_solution

import dede as dd


def test_add1():
    # TODO: relax, change constraints/edit hyperparameters, make a more lenient threshold
    N, M = 10, 10
    x = dd.Variable((N, M), integer=True)
    resource_constraints = [x >= 0] + [x[i, :].sum() >= 2 * i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= 3 * j for j in range(M)]
    expr = 0
    for i in range(min(N, M)):
        if i % 2 == 0:
            expr += x[i, i]
        else:
            expr -= x[i, i]
    objective = dd.Maximize(expr)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve()
    print("CVXPY:", result_cvxpy)

    prob = dd.Problem(objective, resource_constraints, demand_constraints)
    result_dede = prob.solve(num_cpus=2, solver=dd.GUROBI, rho=0.5, num_iter=25, **GUROBI_OPTS)
    print("DeDe:", result_dede)
    print(x.value)

    assert check_solution(result_dede, result_cvxpy, objective)
    print("=== Passed ILP ADD test #1 ===")


def test_add2():
    N, M = 10, 10
    x = dd.Variable((N, M), integer=True)
    resource_constraints = [x >= 0] + [x[i, :].sum() <= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= j for j in range(M)]

    # pull two elements
    objective = dd.Maximize(x[N - 1, M - 1] + x[N // 2, M // 2])

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve()
    print("CVXPY:", result_cvxpy)

    prob = dd.Problem(objective, resource_constraints, demand_constraints)
    result_dede = prob.solve(num_cpus=2, solver=dd.GUROBI, rho=0.5, num_iter=30, **GUROBI_OPTS)
    print("DeDe:", result_dede)

    assert check_solution(result_dede, result_cvxpy, objective)
    print("=== Passed ILP ADD test #2 ===")


def test_constant1():
    N, M = 5, 5
    x = dd.Variable((N, M), integer=True)
    resource_constraints = [x >= 0] + [x[i, :].sum() <= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= j for j in range(M)]

    objective = dd.Maximize(2)

    prob = dd.Problem(objective, resource_constraints, demand_constraints)
    result_dede = prob.solve(num_cpus=2, solver=dd.GUROBI, rho=1, num_iter=5, **GUROBI_OPTS)
    print("DeDe:", result_dede)

    assert check_solution(result_dede, 2, objective)
    print("=== Passed ILP CONSTANT test #1 ===")


def test_constant2():
    N, M = 10, 10
    x = dd.Variable((N, M), integer=True)
    resource_constraints = [x >= 0] + [x[i, :].sum() <= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= j for j in range(M)]

    objective = dd.Maximize(x[N - 1, M - 1] + 2)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve()
    print("CVXPY:", result_cvxpy)

    prob = dd.Problem(objective, resource_constraints, demand_constraints)
    result_dede = prob.solve(num_cpus=2, solver=dd.GUROBI, rho=0.5, num_iter=20, **GUROBI_OPTS)
    print("DeDe:", result_dede)

    assert check_solution(result_dede, result_cvxpy, objective)
    print("=== Passed ILP CONSTANT test #2 ===")


def test_sum1():
    N, M = 5, 5
    x = dd.Variable((N, M), integer=True)
    resource_constraints = [x[i, :].sum() >= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= j for j in range(M)]
    objective = dd.Maximize(dd.sum(x))

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve()
    print("CVXPY:", result_cvxpy)

    prob = dd.Problem(objective, resource_constraints, demand_constraints)
    result_dede = prob.solve(num_cpus=2, solver=dd.GUROBI, rho=1, num_iter=30, **GUROBI_OPTS)
    print("DeDe:", result_dede)

    assert check_solution(result_dede, result_cvxpy, objective)
    print("=== Passed ILP SUM test #1 ===")


def test_sum2():
    N, M = 5, 5
    x = dd.Variable((N, M), integer=True)
    resource_constraints = [x[i, :].sum() <= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= 1 for j in range(M)]
    objective = dd.Maximize(dd.sum(x))

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve()
    print("CVXPY:", result_cvxpy)

    prob = dd.Problem(objective, resource_constraints, demand_constraints)
    result_dede = prob.solve(num_cpus=2, solver=dd.GUROBI, rho=1, num_iter=7, **GUROBI_OPTS)
    print("DeDe:", result_dede)

    assert check_solution(result_dede, result_cvxpy, objective)
    print("=== Passed ILP SUM test #2 ===")


def test_multiply1():
    N, M = 10, 10
    x = dd.Variable((N, M), integer=True)
    resource_constraints = [x >= 0] + [x[i, :].sum() >= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= j for j in range(M)]
    w = np.empty((N, M))
    for i in range(N):
        for j in range(M):
            w[i][j] = i - j
    objective = dd.Maximize(dd.sum(dd.multiply(x, w)))

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve()
    print("CVXPY:", result_cvxpy)

    prob = dd.Problem(objective, resource_constraints, demand_constraints)
    result_dede = prob.solve(num_cpus=2, solver=dd.GUROBI, rho=1, num_iter=14, **GUROBI_OPTS)
    print("DeDe:", result_dede)

    assert check_solution(result_dede, result_cvxpy, objective)
    print("=== Passed ILP MULTIPLY test #1 ===")


def test_multiply2():
    N, M = 10, 10
    x = dd.Variable((N, M), integer=True)
    resource_constraints = [x >= 0] + [x[i, :].sum() >= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= j for j in range(M)]
    w = np.empty((N, M))
    for i in range(N):
        for j in range(M):
            w[i][j] = i * j / (i + j + 1)
    objective = dd.Maximize(dd.sum(dd.multiply(x, w)))

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve()
    print("CVXPY:", result_cvxpy)

    prob = dd.Problem(objective, resource_constraints, demand_constraints)
    result_dede = prob.solve(num_cpus=2, solver=dd.GUROBI, rho=0.1, num_iter=25, **GUROBI_OPTS)
    print("DeDe:", result_dede)

    assert check_solution(result_dede, result_cvxpy, objective)
    print("=== Passed ILP MULTIPLY test #2 ===")


def test_boolean():
    N, M = 10, 10
    x = dd.Variable((N, M), boolean=True)

    resource_constraints = [x[i, :].sum() >= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= j for j in range(M)]
    objective = dd.Maximize(dd.sum(x))

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve(solver=cp.ECOS_BB)
    print("CVXPY:", result_cvxpy)

    prob = dd.Problem(objective, resource_constraints, demand_constraints)
    result_dede = prob.solve(num_cpus=2, solver=dd.GUROBI, rho=10, num_iter=15, **GUROBI_OPTS)
    print("DeDe:", result_dede)

    assert check_solution(result_dede, result_cvxpy, objective)
    print("=== Passed BOOLEAN LP test ===")


if __name__ == "__main__":
    test_add1()
    test_add2()

    test_constant1()
    test_constant2()

    test_sum1()
    test_sum2()

    test_multiply1()
    test_multiply2()

    test_boolean()
