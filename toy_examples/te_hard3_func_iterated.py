#!/usr/bin/env python3

import dede as dd
import numpy as np
from dede.constraints_utils import func
import random
import matplotlib.pyplot as plt

random.seed(25)
def opt():
    index = []
    dede_list = []
    cvxpy_list = []
    for size in range(2, 24):
        print("Size: ", size)
        index.append(size)
        np.set_printoptions(precision=3, suppress=True)
        N, M = size, size

        # Create allocation variables
        x = dd.Variable((N, M), nonneg=True)

        # Create parameters
        param = dd.Parameter(N, value=[round(random.uniform(1, 3), 1) for i in range(N)])

        # Create the constraints
        resource_constraints = [*func([x.sum(1) >= np.arange(N), x[0,0] + x[0,1] >= 0])]
        demand_constraints = [*func(x.sum(0) <= 5)]

        # Create an objective
        matrix = []
        for i in range(N):
            row = []
            for i in range(M):
                row.append(random.randint(1, N + 10))
            matrix.append(row)
        w = np.array(matrix)
        #w = np.array([[2, 1, 0], [5, 10, 0], [10, 0, 10]])
        objective = dd.Maximize(dd.sum(dd.multiply(x, w)))

        # Construct the problem
        prob = dd.Problem(objective, resource_constraints, demand_constraints)

        # Solve the problem with DeDe on 4 CPU cores
        result_dede = prob.solve(num_cpus=4, solver=dd.ECOS, num_iter=20)
        print(f'dede result: {result_dede}')
        dede_list.append(result_dede)
        # print(f'dede solution:\n{x.value}')

        # Solve the problem with cvxpy
        result_cvxpy = prob.solve(enable_dede=False)
        print(f'cvxpy result: {result_cvxpy}')
        cvxpy_list.append(result_cvxpy)
        print(f'cvxpy solution:\n{x.value}')
    plt.plot(index, dede_list, label='DEDE')
    plt.plot(index, cvxpy_list, label='CVXPY')
    plt.xlabel("Problem Size (N = M)")
    plt.ylabel("Objective Value")
    plt.title("Performance of DEDE vs CVXPY")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("te_hard3_func_result_42.png")


if __name__ == '__main__':
    opt()