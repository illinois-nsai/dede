# #!/usr/bin/env python3

# import dede as dd
# import numpy as np
# from dede.constraints_utils import func
# import random
# import matplotlib.pyplot as plt

# random.seed(25)
# def opt():
#     index = []
#     dede_list = []
#     cvxpy_list = []
#     for size in range(2, 25):
#         print("Size: ", size)
#         index.append(size)
#         np.set_printoptions(precision=3, suppress=True)
#         N, M = size, size

#         # Create allocation variables
#         x = dd.Variable((N, M), nonneg=True)

#         # Create parameters
#         param = dd.Parameter(N, value=[round(random.uniform(1, 3), 1) for i in range(N)])

#         # Create the constraints
#         resource_constraints = [x.sum(1) >= np.arange(N), x[0,0] + x[0,1] >= 0]
#         demand_constraints = [x.sum(0) <= 5]

#         # Create an objective
#         matrix = []
#         for i in range(N):
#             row = []
#             for i in range(M):
#                 row.append(random.randint(1, N + 10))
#             matrix.append(row)
#         w = np.array(matrix)
#         #w = np.array([[2, 1, 0], [5, 10, 0], [10, 0, 10]])
#         objective = dd.Maximize(dd.sum(dd.multiply(x, w)))

#         # Construct the problem
#         prob = dd.Problem(objective, resource_constraints, demand_constraints)

#         # Solve the problem with DeDe on 4 CPU cores
#         result_dede = prob.solve(num_cpus=4, solver=dd.ECOS, num_iter=20)
#         print(f'dede result: {result_dede}')
#         dede_list.append(result_dede)
#         # print(f'dede solution:\n{x.value}')

#         # Solve the problem with cvxpy
#         result_cvxpy = prob.solve(enable_dede=False)
#         print(f'cvxpy result: {result_cvxpy}')
#         cvxpy_list.append(result_cvxpy)
#         print(f'cvxpy solution:\n{x.value}')
#     plt.plot(index, dede_list, label='DEDE')
#     plt.plot(index, cvxpy_list, label='CVXPY')
#     plt.xlabel("Problem Size (N = M)")
#     plt.ylabel("Objective Value")
#     plt.title("Performance of DEDE vs CVXPY")
#     plt.legend()
#     plt.grid(True)
#     plt.tight_layout()
#     plt.savefig("te_hard3_result_42.png")


# if __name__ == '__main__':
#     opt()


#!/usr/bin/env python3

import dede as dd
import numpy as np
#from dede.constraints_utils import func
import random
import matplotlib.pyplot as plt
import cvxpy as cp
import time

random.seed(25)
log_path = "te_hard3.log"
def opt():
    index = []
    dede_list = []
    cvxpy_list = []
    for size in range(100, 1001, 100):
        index.append(size)
        np.set_printoptions(precision=3, suppress=True)
        N, M = size, size
        #dede
        # Create allocation variables
        start_dede = time.time()
        x = dd.Variable((N, M), nonneg=True)
        
        # Create parameters
        param = dd.Parameter(N, value=[round(random.uniform(1, 2), 1) for i in range(N)])
        
        # Create the constraints
        resource_constraints = [x[i, :].sum() >= i for i in range(N)]
        resource_constraints.append(x[0, 0] + x[0, 1] >= 0)

        demand_constraints = [x[:, j].sum() <= size * 3 for j in range(M)]
        
        # Create an objective
        w = np.random.randint(1, 102, size=(N, M))
        #w = np.array([[2, 1, 0], [5, 10, 0], [10, 0, 10]])
        objective = dd.Maximize(dd.sum(dd.multiply(x, w)))
        
        # Construct the problem
        prob = dd.Problem(objective, resource_constraints, demand_constraints)

        # Solve the problem with DeDe on 4 CPU cores
        
        result_dede = prob.solve(num_cpus=4, solver=dd.SCS, num_iter=20,verbose=True)
        end_dede = time.time()
        dede_time = end_dede - start_dede
        print(f'dede result: {result_dede}')
        dede_list.append(result_dede)
        # print(f'dede solution:\n{x.value}')


        # CVXPY
        start_cvx = time.time()
        xc = cp.Variable((N, M), nonneg=True)
        paramc = np.array([round(random.uniform(1, 2), 1) for i in range(N)])
        resource_constraintsc = [cp.sum(xc, axis = 1) >= np.arange(size, size + size), xc[0,0] + xc[0,1] >= 0]
        demand_constraintsc = [cp.sum(xc, axis=0) <= size * 3]
        constraints = resource_constraintsc + demand_constraintsc
        objectivec = cp.Maximize(cp.sum(cp.multiply(xc, w)))


        # Solve the problem with cvxpy
        prob2 = cp.Problem(objectivec, constraints)
        
        result_cvxpy = prob2.solve(solver=cp.SCS)
        end_cvx = time.time()
        cvxpy_time = end_cvx - start_cvx
        #result_cvxpy = prob.solve(enable_dede=False)
        print(f'cvxpy result: {result_cvxpy}')
        cvxpy_list.append(result_cvxpy)
        print(f'cvxpy solution:\n{xc.value}')
        with open(log_path, "a") as f:
            f.write(f"Size: {size}\n")
            f.write(f"  DEDE Value: {result_dede:.4f}, Time: {dede_time:.4f} sec\n")
            f.write(f"  CVXPY Value: {result_cvxpy:.4f}, Time: {cvxpy_time:.4f} sec\n\n")
    plt.plot(index, dede_list, label='DEDE')
    plt.plot(index, cvxpy_list, label='CVXPY')
    for i, val in zip(index, dede_list):
        plt.text(i, val, f'{val:.1f}', ha='center', va='bottom', fontsize=8, color='blue')
    for i, val in zip(index, cvxpy_list):
        plt.text(i, val, f'{val:.1f}', ha='center', va='top', fontsize=8, color='green')
    plt.xlabel("Size")
    plt.ylabel("Value")
    plt.title("DEDE vs CVXPY")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("te_hard3_result.png")


if __name__ == '__main__':
    opt()