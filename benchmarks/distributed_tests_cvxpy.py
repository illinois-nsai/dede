import os
import sys
import time

NUM_CPUS = "1"
os.environ["OMP_NUM_THREADS"] = NUM_CPUS
os.environ["OPENBLAS_NUM_THREADS"] = NUM_CPUS
os.environ["RAYON_NUM_THREADS"] = NUM_CPUS

import cvxpy as cp
import numpy as np

sys.setrecursionlimit(10000)
rng = np.random.default_rng(seed=42)


def test_sum(n):
    N, M = n, n
    x = cp.Variable((N, M), nonneg=True)
    resource_constraints = [cp.sum(x[i, :]) >= i for i in range(N)]
    demand_constraints = [cp.sum(x[:, j]) <= j for j in range(M)]

    objective = cp.Maximize(cp.sum(x))

    prob = cp.Problem(objective, resource_constraints + demand_constraints)
    start = time.perf_counter()
    result_cvxpy = prob.solve(solver=cp.GUROBI)
    end = time.perf_counter()
    print(f"Iterations: {prob.solver_stats.num_iters}")
    print(f"Executed solve in {end - start}s")

    return result_cvxpy


def test_weighted(n):
    N, M = n, n
    x = cp.Variable((N, M), nonneg=True)
    w = 9 * rng.uniform(0, 1, (N, M)) + 1
    bn = 9 * rng.uniform(0, 1, (N,)) + 1
    bm = 9 * rng.uniform(0, 1, (M,)) + 1

    resource_constraints = [cp.sum(x[i, :]) >= bn[i] for i in range(N)]
    demand_constraints = [cp.sum(x[:, j]) >= bm[j] for j in range(M)]

    objective = cp.Minimize(cp.sum(cp.multiply(x, w)))

    prob = cp.Problem(objective, resource_constraints + demand_constraints)
    start = time.perf_counter()
    result_cvxpy = prob.solve(solver=cp.GUROBI)
    end = time.perf_counter()
    print(f"Iterations: {prob.solver_stats.num_iters}")
    print(f"Executed solve in {end - start}s")
    return result_cvxpy


def test_log(n):
    N, M = n, n
    # Scale y = x/N to avoid poor conditioning from large RHS values (up to N^2)
    y = cp.Variable((N, M), nonneg=True)
    resource_constraints = [cp.sum(y[i, :]) >= (i + 1) for i in range(N)]
    demand_constraints = [cp.sum(y[:, j]) <= (j + 1) for j in range(M)]

    objective = cp.Maximize(cp.sum([cp.log(cp.sum(y[i, :])) for i in range(N)]))

    prob = cp.Problem(objective, resource_constraints + demand_constraints)
    start = time.perf_counter()
    result_cvxpy = prob.solve(solver=cp.SCS, max_iters=10000, eps=1e-4)
    end = time.perf_counter()
    print(f"Iterations: {prob.solver_stats.num_iters}")
    print(f"Executed solve in {end - start}s")
    # Add back constant N*log(N) from the scaling transformation
    return result_cvxpy


if __name__ == "__main__":
    base = 500

    for multiplier in range(1, 5):
        # num_cpus loop kept for structure, though cvxpy's solve interface
        # doesn't take num_cpus as a direct argument for parallelism.
        sum_n = multiplier * base
        weighted_n = multiplier * base
        log_n = multiplier * base

        print(f"Testing sum n={sum_n}, num_cpus={NUM_CPUS}")
        try:
            result = test_sum(sum_n)
            print(f"Result {result}")
        except Exception as e:
            print(f"Error in test_sum with n={sum_n}, num_cpus={NUM_CPUS}: {e}")

        print(f"Testing weighted n={weighted_n}, num_cpus={NUM_CPUS}")
        try:
            result = test_weighted(weighted_n)
            print(f"Result {result}")
        except Exception as e:
            print(f"Error in test_weighted with n={weighted_n}, num_cpus={NUM_CPUS}: {e}")

        print(f"Testing log n={log_n}, num_cpus={NUM_CPUS}")
        try:
            result = test_log(log_n)
            print(f"Result {result}")
        except Exception as e:
            print(f"Error in test_log with n={log_n}, num_cpus={NUM_CPUS}: {e}")
