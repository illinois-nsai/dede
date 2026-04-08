import argparse
import os
import sys

NUM_CPUS = "4"

os.environ["OMP_NUM_THREADS"] = NUM_CPUS
os.environ["MKL_NUM_THREADS"] = NUM_CPUS
os.environ["OPENBLAS_NUM_THREADS"] = NUM_CPUS
os.environ["VECLIB_MAXIMUM_THREADS"] = NUM_CPUS
os.environ["NUMEXPR_NUM_THREADS"] = NUM_CPUS

import cvxpy as cp
import numpy as np

sys.setrecursionlimit(10000)


def test_sum(n):
    N, M = n, n
    x = cp.Variable((N, M), nonneg=True)
    resource_constraints = [cp.sum(x[i, :]) >= i for i in range(N)]
    demand_constraints = [cp.sum(x[:, j]) <= j for j in range(M)]

    objective = cp.Maximize(cp.sum(x))

    prob = cp.Problem(objective, resource_constraints + demand_constraints)
    # CVXPY solvers generally manage threading internally via their own libraries (like OpenBLAS)
    result_cvxpy = prob.solve(solver=cp.ECOS)
    return result_cvxpy, x


def test_weighted(n):
    N, M = n, n
    x = cp.Variable((N, M), nonneg=True)
    w = 9 * np.random.uniform(0, 1, (N, M)) + 1
    bn = 9 * np.random.uniform(0, 1, (N,)) + 1
    bm = 9 * np.random.uniform(0, 1, (M,)) + 1

    resource_constraints = [cp.sum(x[i, :]) >= bn[i] for i in range(N)]
    demand_constraints = [cp.sum(x[:, j]) >= bm[j] for j in range(M)]

    objective = cp.Minimize(cp.sum(cp.multiply(x, w)))

    prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = prob.solve(solver=cp.ECOS)
    return result_cvxpy, x


def test_log(n):
    N, M = n, n
    x = cp.Variable((N, M), nonneg=True)
    resource_constraints = [cp.sum(x[i, :]) >= (i + 1) * M for i in range(N)]
    demand_constraints = [cp.sum(x[:, j]) <= (j + 1) * N for j in range(M)]

    # Using cp.sum on a list comprehension to match original structure
    objective = cp.Maximize(cp.sum([cp.log(cp.sum(x[i, :])) for i in range(N)]))

    prob = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = prob.solve(solver=cp.SCS)
    return result_cvxpy, x


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

    for multiplier in range(31):
        # num_cpus loop kept for structure, though cvxpy's solve interface
        # doesn't take num_cpus as a direct argument for parallelism.
        sum_n = multiplier * sum_multiplier
        weighted_n = multiplier * weighted_multiplier
        log_n = multiplier * log_multiplier

        print(f"Testing sum n={sum_n}, num_cpus={NUM_CPUS}")
        try:
            result, x = test_sum(sum_n)
            print(f"Result {result}")
            print(f"Variables x: {x.value}")
        except Exception as e:
            print(f"Error in test_sum with n={sum_n}, num_cpus={NUM_CPUS}: {e}")

        print(f"Testing weighted n={weighted_n}, num_cpus={NUM_CPUS}")
        try:
            result, x = test_weighted(weighted_n)
            print(f"Result {result}")
            print(f"Variables x: {x.value}")
        except Exception as e:
            print(f"Error in test_weighted with n={weighted_n}, num_cpus={NUM_CPUS}: {e}")

        print(f"Testing log n={log_n}, num_cpus={NUM_CPUS}")
        try:
            result, x = test_log(log_n)
            print(f"Result {result}")
            print(f"Variables x: {x.value}")
        except Exception as e:
            print(f"Error in test_log with n={log_n}, num_cpus={NUM_CPUS}: {e}")
