import sys

import cvxpy as cp
import numpy as np

import dede as dd

sys.setrecursionlimit(10000)


def test_sum(n, num_cpus):
    N, M = n, n
    x = dd.Variable((N, M), nonneg=True)
    resource_constraints = [x[i, :].sum() >= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= j for j in range(M)]

    objective = dd.Maximize(dd.sum(x))

    result_cvxpy = cp.Problem(objective, resource_constraints + demand_constraints).solve(
        solver=cp.GUROBI
    )

    prob = dd.Problem(objective, resource_constraints, demand_constraints)
    result_dede = prob.solve(ray_address="auto", solver=dd.GUROBI, num_cpus=num_cpus)
    print(result_cvxpy, result_dede)

    return result_dede


def test_weighted(n, num_cpus):
    N, M = n, n
    x = dd.Variable((N, M), nonneg=True)
    w = 9 * np.random.uniform(0, 1, (N, M)) + 1
    bn = 9 * np.random.uniform(0, 1, (N,)) + 1
    bm = 9 * np.random.uniform(0, 1, (M,)) + 1

    resource_constraints = [x[i, :].sum() >= bn[i] for i in range(N)]
    demand_constraints = [x[:, j].sum() >= bm[j] for j in range(M)]

    objective = dd.Minimize(dd.sum(dd.multiply(x, w)))

    result_cvxpy = cp.Problem(objective, resource_constraints + demand_constraints).solve(
        solver=cp.GUROBI
    )

    prob = dd.Problem(objective, resource_constraints, demand_constraints)
    result_dede = prob.solve(ray_address="auto", num_cpus=num_cpus, solver=dd.GUROBI)

    print(result_cvxpy, result_dede)

    return result_dede


def test_log(n, num_cpus):
    N, M = n, n
    x = cp.Variable((N, M), nonneg=True)
    resource_constraints = [x[i, :].sum() >= (i + 1) * M for i in range(N)]
    demand_constraints = [x[:, j].sum() <= (j + 1) * N for j in range(M)]
    objective = cp.Maximize(sum([cp.log(cp.sum(x[i, :])) for i in range(N)]))

    result_dede = dd.Problem(objective, resource_constraints, demand_constraints).solve(
        num_cpus=num_cpus, solver=dd.SCS, ray_address="auto"
    )

    result_cvxpy = cp.Problem(objective, resource_constraints + demand_constraints).solve(
        solver=cp.SCS
    )
    print(result_cvxpy, result_dede)

    return result_dede


if __name__ == "__main__":
    sum_multiplier = 80
    weighted_multiplier = 30
    log_multiplier = 10
    for multiplier in [5]:
        for num_cpus in [8]:
            sum_n = multiplier * sum_multiplier
            weighted_n = multiplier * weighted_multiplier
            log_n = multiplier * log_multiplier

            print(f"Testing sum n={sum_n}, num_cpus={num_cpus}")
            try:
                result = test_sum(sum_n, num_cpus)
                print(f"Result {result}")
            except Exception as e:
                print(f"Error in test_sum with n={sum_n}, num_cpus={num_cpus}: {e}")

            # print(f"Testing weighted n={weighted_n}, num_cpus={num_cpus}")
            # try:
            #     result = test_weighted(weighted_n, num_cpus)
            #     print(f"Result {result}")
            # except Exception as e:
            #     print(f"Error in test_weighted with n={weighted_n}, num_cpus={num_cpus}: {e}")

            # print(f"Testing log n={log_n}, num_cpus={num_cpus}")
            # try:
            #     result = test_log(log_n, num_cpus)
            #     print(f"Result {result}")
            # except Exception as e:
            #     print(f"Error in test_log with n={log_n}, num_cpus={num_cpus}: {e}")
