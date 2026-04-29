import sys

import numpy as np

import dede as dd

sys.setrecursionlimit(10000)

rng = np.random.default_rng(seed=42)


def test_sum(n, num_cpus):
    N, M = n, n
    x = dd.Variable((N, M), nonneg=True)
    resource_constraints = [x[i, :].sum() >= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= j for j in range(M)]

    objective = dd.Maximize(dd.sum(x))

    prob = dd.Problem(objective, resource_constraints, demand_constraints)
    result_dede = prob.solve(ray_address="auto", solver=dd.GUROBI, num_cpus=num_cpus)

    return result_dede


def test_weighted(n, num_cpus):
    N, M = n, n
    x = dd.Variable((N, M), nonneg=True)
    w = 9 * rng.uniform(0, 1, (N, M)) + 1
    bn = 9 * rng.uniform(0, 1, (N,)) + 1
    bm = 9 * rng.uniform(0, 1, (M,)) + 1

    resource_constraints = [x[i, :].sum() >= bn[i] for i in range(N)]
    demand_constraints = [x[:, j].sum() >= bm[j] for j in range(M)]

    objective = dd.Minimize(dd.sum(dd.multiply(x, w)))

    prob = dd.Problem(objective, resource_constraints, demand_constraints)
    result_dede = prob.solve(ray_address="auto", num_cpus=num_cpus, solver=dd.GUROBI)

    return result_dede


def test_log(n, num_cpus):
    N, M = n, n
    # Scale y = x/N to avoid poor conditioning from large RHS values (up to N^2)
    y = dd.Variable((N, M), nonneg=True)
    resource_constraints = [y[i, :].sum() >= (i + 1) for i in range(N)]
    demand_constraints = [y[:, j].sum() <= (j + 1) for j in range(M)]

    objective = dd.Maximize(dd.sum([dd.log(dd.sum(y[i, :])) for i in range(N)]))

    prob = dd.Problem(objective, resource_constraints, demand_constraints)
    result_dede = prob.solve(ray_address="auto", num_cpus=num_cpus, solver=dd.SCS)

    # Add back constant N*log(N) from the scaling transformation
    return result_dede


if __name__ == "__main__":
    base = 500
    for multiplier in range(1, 5):
        for num_cpus in [64, 32, 16, 8, 4, 2, 1]:
            sum_n = multiplier * base
            weighted_n = multiplier * base
            log_n = multiplier * base

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
