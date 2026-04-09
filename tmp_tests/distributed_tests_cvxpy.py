import argparse
import sys

import cvxpy as cp
import numpy as np

sys.setrecursionlimit(10000)


def test_sum(n, num_cpus):
    N, M = n, n
    x = cp.Variable((N, M), nonneg=True)
    resource_constraints = [cp.sum(x[i, :]) >= i for i in range(N)]
    demand_constraints = [cp.sum(x[:, j]) <= j for j in range(M)]

    objective = cp.Maximize(cp.sum(x))

    prob = cp.Problem(objective, resource_constraints + demand_constraints)
    # CVXPY solvers generally manage threading internally via their own libraries (like OpenBLAS)
    result_cvxpy = prob.solve(solver=cp.GUROBI, Threads=num_cpus, verbose=True)
    return result_cvxpy


def test_weighted(n, num_cpus):
    N, M = n, n
    x = cp.Variable((N, M), nonneg=True)
    w = 9 * np.random.uniform(0, 1, (N, M)) + 1
    bn = 9 * np.random.uniform(0, 1, (N,)) + 1
    bm = 9 * np.random.uniform(0, 1, (M,)) + 1

    resource_constraints = [cp.sum(x[i, :]) >= bn[i] for i in range(N)]
    demand_constraints = [cp.sum(x[:, j]) >= bm[j] for j in range(M)]

    objective = cp.Minimize(cp.sum(cp.multiply(x, w)))

    prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = prob.solve(solver=cp.GUROBI, Threads=num_cpus, verbose=True)
    return result_cvxpy


def test_log(n, num_cpus):
    N, M = n, n
    x = cp.Variable((N, M), nonneg=True)
    t = cp.Variable(N)

    resource_constraints = [cp.sum(x[i, :]) >= (i + 1) * M for i in range(N)]
    demand_constraints = [cp.sum(x[:, j]) <= (j + 1) * N for j in range(M)]
    log_constraints = [t[i] <= cp.log(cp.sum(x[i, :])) for i in range(N)]

    objective = cp.Maximize(cp.sum(t))
    prob = cp.Problem(objective, resource_constraints + demand_constraints + log_constraints)
    result_cvxpy = prob.solve(solver=cp.GUROBI, Threads=num_cpus, verbose=True)
    return result_cvxpy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CVXPY benchmarks")
    parser.add_argument(
        "--num_cpus",
        type=int,
        default=1,
        help="Number of CPUs to use",
    )
    sum_multiplier = 80
    weighted_multiplier = 30
    log_multiplier = 10

    for multiplier in range(5, 31):
        for num_cpus in [64]:
            # num_cpus loop kept for structure, though cvxpy's solve interface
            # doesn't take num_cpus as a direct argument for parallelism.
            sum_n = multiplier * sum_multiplier
            weighted_n = multiplier * weighted_multiplier
            log_n = multiplier * log_multiplier

            print(f"Testing sum n={sum_n}, num_cpus={num_cpus}")
            try:
                result = test_sum(sum_n, num_cpus)
                print(f"Result {result}")
            except Exception as e:
                print(f"Error in test_sum with n={sum_n}, num_cpus={num_cpus}: {e}")

            print(f"Testing weighted n={weighted_n}, num_cpus={num_cpus}")
            try:
                result = test_weighted(weighted_n, num_cpus)
                print(f"Result {result}")
            except Exception as e:
                print(f"Error in test_weighted with n={weighted_n}, num_cpus={num_cpus}: {e}")

            print(f"Testing log n={log_n}, num_cpus={num_cpus}")
            try:
                result = test_log(log_n, num_cpus)
                print(f"Result {result}")
            except Exception as e:
                print(f"Error in test_log with n={log_n}, num_cpus={num_cpus}: {e}")
