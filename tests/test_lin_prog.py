#!/usr/bin/env python3

import dede as dd
import cvxpy as cp
import numpy as np
from scipy.optimize import minimize
import math


def add1():
    N, M = 100, 100
    x = dd.Variable((N, M), nonneg=True)
    resource_constraints = [x[i, :].sum() >= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= j for j in range(M)]
    expr = 0
    for i in range(min(N, M)):
        if i % 2 == 0:
            expr += x[i, i]
        else:
            expr -= x[i, i]
    objective = dd.Maximize(expr)

    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=4, solver=dd.ECOS, rho=0.075, num_iter=19)
    print("DeDe:", result_dede)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve()
    print("CVXPY:", result_cvxpy)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01)
    print('=== Passed ADD test #1 ===')


def add2():
    N, M = 100, 100
    x = dd.Variable((N, M), nonneg=True)
    resource_constraints = [x[i, :].sum() <= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= j for j in range(M)]
    
    # pull two elements
    objective = dd.Maximize(x[N-1, M-1] + x[N//2, M//2])

    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=4, solver=dd.ECOS, rho=0.1, num_iter=20)
    print("DeDe:", result_dede)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve()
    print("CVXPY:", result_cvxpy)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01)
    print('=== Passed ADD test #2 ===')


def add_zero():
    N, M = 3, 3
    x = dd.Variable((N, M), nonneg=True)
    resource_constraints = [x[i, :].sum() <= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= j for j in range(M)]
    
    # pull two elements
    objective = dd.Maximize(0 * x[0, 0] + 0 * x[1, 1] + 0 * x[2, 2])

    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=4, solver=dd.ECOS, rho=0.1, num_iter=20)
    print("DeDe:", result_dede)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve()
    print("CVXPY:", result_cvxpy)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01)
    print('=== Passed ADD zero test ===') 


def sum1():
    N, M = 100, 100
    x = dd.Variable((N, M), nonneg=True)
    resource_constraints = [x[i, :].sum() >= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= j for j in range(M)]
    objective = dd.Maximize(dd.sum(x))

    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=4, solver=dd.ECOS, rho=1, num_iter=5)
    print("DeDe:", result_dede)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve()
    print("CVXPY:", result_cvxpy)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01)
    print('=== Passed SUM test #1 ===')


def sum2():
    N, M = 100, 100
    x = dd.Variable((N, M), nonneg=True)
    resource_constraints = [x[i, :].sum() <= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= 1 for j in range(M)]
    objective = dd.Maximize(dd.sum(x))

    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=4, solver=dd.ECOS, rho=20, num_iter=7)
    print("DeDe:", result_dede)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve()
    print("CVXPY:", result_cvxpy)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01)
    print('=== Passed SUM test #2 ===')


def multiply1():
    N, M = 100, 100
    x = dd.Variable((N, M), nonneg=True)
    resource_constraints = [x[i, :].sum() >= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= j for j in range(M)]
    w = np.empty((N, M))
    for i in range(N):
        for j in range(M):
            w[i][j] = i - j
    objective = dd.Maximize(dd.sum(dd.multiply(x, w)))

    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=4, solver=dd.ECOS, rho=125, num_iter=60)
    print("DeDe:", result_dede)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve()
    print("CVXPY:", result_cvxpy)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01, abs_tol=0.1)
    print('=== Passed MULTIPLY test #1 ===')


def multiply2():
    N, M = 100, 100
    x = dd.Variable((N, M), nonneg=True)
    resource_constraints = [x[i, :].sum() >= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= j for j in range(M)]
    w = np.empty((N, M))
    for i in range(N):
        for j in range(M):
            w[i][j] = i * j / (i + j + 1)
    objective = dd.Maximize(dd.sum(dd.multiply(x, w)))

    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=4, solver=dd.ECOS, rho=10, num_iter=40)
    print("DeDe:", result_dede)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve()
    print("CVXPY:", result_cvxpy)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01, abs_tol=0.1)
    print('=== Passed MULTIPLY test #2 ===')


def multiply_zero():
    N, M = 3, 4
    x = dd.Variable((N, M), nonneg=True)
    resource_constraints = [x[i, :].sum() >= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= j for j in range(M)]
    w = np.zeros((N, M))
    w[0][2] = 1
    objective = dd.Maximize(dd.sum(dd.multiply(x, w)))

    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=4, solver=dd.ECOS, rho=1, num_iter=10)
    print("DeDe:", result_dede)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve()
    print("CVXPY:", result_cvxpy)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01, abs_tol=0.1)
    print('=== Passed MULTIPLY zero test ===')


def log():
    N, M = 10, 10
    x = dd.Variable((N, M), nonneg=True)

    # nonnegative log values: x >= 1
    resource_constraints = [x >= 1] + [x[i, :].sum() <= (i + 1) * M for i in range(N)]
    demand_constraints = [x[:, j].sum() <= (j + 1) * N for j in range(M)]
    
    # write in separable form, dd.sum(x) fails
    expr = dd.sum([dd.sum(dd.log(x[i, :])) for i in range(N)])
    objective = dd.Maximize(expr)

    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=4, solver=dd.ECOS, rho=0.08, num_iter=30)
    print("DeDe:", result_dede)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve()
    print("CVXPY:", result_cvxpy)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01)
    print('=== Passed LOG test ===') 


def log_weighted():
    N, M = 10, 10
    x = dd.Variable((N, M), nonneg=True)

    # nonnegative log values: x >= 1
    resource_constraints = [x >= 1] + [x[i, :].sum() <= (i + 1) * M for i in range(N)]
    demand_constraints = [x[:, j].sum() <= (j + 1) * N for j in range(M)]
    
    # write in separable form, dd.sum(x) fails
    w = np.empty((N, M))
    for i in range(N):
        for j in range(M):
            w[i][j] = i + j
    expr = dd.sum(dd.multiply(w, dd.log(x)))
    objective = dd.Maximize(expr)

    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=4, solver=dd.ECOS, rho=1, num_iter=50)
    print("DeDe:", result_dede)

    cvxpy_prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve()
    print("CVXPY:", result_cvxpy)

    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.01)
    print('=== Passed LOG weighted test ===')  


def quadratic():
    N, M = 10, 10
    x = dd.Variable((N, M), nonneg=True)

    resource_constraints = [x[i, :].sum() >= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= j for j in range(M)]
    objective = dd.Minimize(dd.sum_squares(x))
    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    result_dede = prob.solve(num_cpus=4, solver=dd.ECOS, rho=12, num_iter=7)
    print("DeDe:", result_dede)
    
    # internal bug with cvxpy: cp.sum_squares fails
    '''
    cvxpy_prob = cp.Problem(cp.Minimize(cp.sum_squares(x)), resource_constraints + demand_constraints)
    result_cvxpy = cvxpy_prob.solve()
    print("CVXPY:", result_cvxpy)
    '''

    # use scipy.optimize instead

    # Flattened variable: x = x.flatten()
    def objective(x):
        return np.sum(x**2)

    # Constraints
    constraints = []

    # Resource constraints: sum over each row ≥ i
    for i in range(N):
        idx = [i * M + j for j in range(M)]
        coeff = np.zeros(N * M)
        coeff[idx] = 1
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, coeff=coeff, b=i: np.dot(coeff, x) - b
        })

    # Demand constraints: sum over each column ≤ j
    for j in range(M):
        idx = [i * M + j for i in range(N)]
        coeff = np.zeros(N * M)
        coeff[idx] = 1
        constraints.append({
            'type': 'ineq',
            'fun': lambda x, coeff=coeff, b=j: b - np.dot(coeff, x)
        })

    # Bounds: x[i, j] ≥ 0
    bounds = [(0, None)] * (N * M)

    # Initial guess (small positive values)
    x0 = np.ones(N * M)

    # Solve
    result = minimize(objective, x0, method='SLSQP', bounds=bounds, constraints=constraints)

    # Reshape result
    x_opt = result.x.reshape((N, M))
    print("Optimal x:\n", np.round(x_opt, 4))
    print("Objective value:", round(result.fun, 4))

    assert math.isclose(result_dede, result.fun, rel_tol=0.01)
    print('=== Passed QUADRATIC test ===')



if __name__ == '__main__':
    add1()
    add2()
    add_zero()

    sum1()
    sum2()

    multiply1()
    multiply2()
    multiply_zero()

    log()
    log_weighted()

    quadratic()