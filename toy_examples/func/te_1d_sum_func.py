#!/usr/bin/env python3

import dede as dd
import numpy as np
from dede.constraints_utils import func


def opt():
    np.set_printoptions(precision=3, suppress=True)
    N, M = 4, 1

    # Create allocation variables
    x = dd.Variable((N, M), nonneg=True)

    # Create parameters
    param = dd.Parameter(N, value=[1.0, 0.5, 1.2, 1.7])

    # Create the constraints
    resource_constraints = func([x[:] >= param[i] for i in range(N)])
    demand_constraints = func([x.sum() <= 4])

    # Create an objective
    w = np.array([[2, 1, 0, 7]])
    objective = dd.Maximize(dd.sum(dd.multiply(x, w)))

    # Construct the problem
    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    # Solve the problem with DeDe on 4 CPU cores
    result_dede = prob.solve(num_cpus=4, solver=dd.ECOS, num_iter=20)
    print(f'dede result: {result_dede}')
    # print(f'dede solution:\n{x.value}')

    # Solve the problem with cvxpy
    result_cvxpy = prob.solve(enable_dede=False)
    print(f'cvxpy result: {result_cvxpy}')
    print(f'cvxpy solution:\n{x.value}')


if __name__ == '__main__':
    opt()

#Results:
# dede: -46.24043853262186
# cvxpy: inf