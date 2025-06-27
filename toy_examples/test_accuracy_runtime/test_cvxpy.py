import dede as dd
import cvxpy as cp
import time
import csv
import os

def test(N, M):
    '''
        DeDe version
    '''
    dede_start = time.time()

    # Create allocation variables
    x = dd.Variable((N, M), nonneg=True)

    # Create the constraints
    resource_constraints = [x[i,:].sum() >= i for i in range(N)]
    demand_constraints = [x[:,j].sum() <= j for j in range(M)]

    # Create an objective
    objective = dd.Minimize(x.sum())

    # Construct the problem
    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    # Solve the problem with DeDe on 16 CPU cores
    dede_result = prob.solve(num_cpus=16, solver=dd.ECOS)

    dede_end = time.time()


    '''
        CVXPY version
    '''
    cvxpy_start = time.time()
    # Create allocation variables
    x = cp.Variable((N, M), nonneg=True)

    # Create the constraints
    resource_constraints = [x[i,:].sum() >= i for i in range(N)]
    demand_constraints = [x[:,j].sum() <= j for j in range(M)]

    constraints = resource_constraints + demand_constraints

    # Create an objective
    objective = cp.Minimize(x.sum())

    # Construct the problem
    prob = cp.Problem(objective, constraints)

    # Solve the problem with CVXPY
    cvxpy_result = prob.solve()

    cvxpy_end = time.time()

    return (dede_result, dede_end - dede_start, cvxpy_result, cvxpy_end - cvxpy_start)

if __name__ == '__main__':
    log_file = "solve_times.csv"
    fieldnames = ["N", "DeDe_result", "DeDe_time", "CVXPY_result", "CVXPY_time"]

    if not os.path.exists(log_file):
        with open(log_file, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

    for N in range(1500, 10001, 1000):
        dede_result, dede_time, cvxpy_result, cvxpy_time = test(N, N)

        with open(log_file, mode='a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writerow({
                "N": N,
                "DeDe_result": dede_result,
                "DeDe_time": dede_time,
                "CVXPY_result": cvxpy_result,
                "CVXPY_time": cvxpy_time
            })