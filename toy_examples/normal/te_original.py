#!/usr/bin/env python3

import dede as dd
import numpy as np
import cvxpy as cp

def opt():
    np.set_printoptions(precision=3, suppress=True)
    N, M = 3, 3

    # Create allocation variables
    x = dd.Variable((N, M), nonneg=True)
    xc = cp.Variable((N, M), nonneg=True)

    # Create parameters
    param = dd.Parameter(N, value=[1.0, 0.5, 1.2])
    paramc = np.array([1.0, 0.5, 1.2])

    # Create the constraints
    resource_constraints = [x[i, :].sum() <= param[i] for i in range(N)]
    demand_constraints = [x[:, j].sum() <= 1 for j in range(M)]
    resource_constraintsc = [cp.sum(xc[i, :]) <= paramc[i] for i in range(N)]
    demand_constraintsc = [cp.sum(xc[:, j]) <= 1 for j in range(M)]
    constraints = resource_constraintsc + demand_constraintsc

    # Create an objective
    w = np.array([[2, 1, 0], [5, 10, 0], [10, 0, 10]])
    objective = dd.Maximize(dd.sum(dd.multiply(x, w)))
    objectivec = cp.Maximize(cp.sum(cp.multiply(xc, w)))

    # Construct the problem
    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    # Solve the problem with DeDe on 4 CPU cores
    result_dede = prob.solve(num_cpus=4, solver=dd.ECOS, num_iter=20)
    print(f'dede result: {result_dede}')
    # print(f'dede solution:\n{x.value}')

    # Solve the problem with cvxpy
    prob2 = cp.Problem(objectivec, constraints)
    result_cvxpy = prob2.solve(solver=cp.ECOS)
    #result_cvxpy = prob.solve(enable_dede=False)
    print(f'cvxpy result: {result_cvxpy}')
    print(f'cvxpy solution:\n{xc.value}')


if __name__ == '__main__':
    opt()