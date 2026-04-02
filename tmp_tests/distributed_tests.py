import sys

import numpy as np

import dede as dd

sys.setrecursionlimit(10000)


def test_sum(n, num_cpus):
    N, M = n, n
    x = dd.Variable((N, M), nonneg=True)
    resource_constraints = [x[i, :].sum() >= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= j for j in range(M)]

    objective = dd.Maximize(dd.sum(x))

    prob = dd.Problem(objective, resource_constraints, demand_constraints)
    result_dede = prob.solve(ray_address="auto", solver=dd.ECOS, num_cpus=num_cpus)


def test_weighted(n, num_cpus):
    N, M = n, n
    x = dd.Variable((N, M), nonneg=True)
    w = 9 * np.random.uniform(0, 1, (N, M)) + 1
    bn = 9 * np.random.uniform(0, 1, (N,)) + 1
    bm = 9 * np.random.uniform(0, 1, (M,)) + 1

    resource_constraints = [x[i, :].sum() >= bn[i] for i in range(N)]
    demand_constraints = [x[:, j].sum() >= bm[j] for j in range(M)]

    objective = dd.Minimize(dd.sum(dd.multiply(x, w)))

    prob = dd.Problem(objective, resource_constraints, demand_constraints)
    result_dede = prob.solve(ray_address="auto", num_cpus=num_cpus, solver=dd.ECOS)


def test_log(n, num_cpus):
    N, M = n, n
    x = dd.Variable((N, M), nonneg=True)
    resource_constraints = [x[i, :].sum() >= (i + 1) * M for i in range(N)]
    demand_constraints = [x[:, j].sum() <= (j + 1) * N for j in range(M)]

    objective = dd.Maximize(dd.sum([dd.log(dd.sum(x[i])) for i in range(N)]))

    prob = dd.Problem(objective, resource_constraints, demand_constraints)
    result_dede = prob.solve(ray_address="auto", num_cpus=num_cpus, solver=dd.SCS)


if __name__ == "__main__":
    for n in np.arange(200, 2001, 200):
        for num_cpus in [1, 2, 4, 8, 16, 32]:
            print(f"Testing sum n={n}, num_cpus={num_cpus}")
            try:
                test_sum(n, num_cpus)
            except Exception as e:
                print(f"Error in test_sum with n={n}, num_cpus={num_cpus}: {e}")

            print(f"Testing weighted n={n}, num_cpus={num_cpus}")
            try:
                test_weighted(n, num_cpus)
            except Exception as e:
                print(f"Error in test_weighted with n={n}, num_cpus={num_cpus}: {e}")

            print(f"Testing log n={n} num_cpus={num_cpus}")
            try:
                test_log(n, num_cpus)
            except Exception as e:
                print(f"Error in test_log with n={n}: {e}")
