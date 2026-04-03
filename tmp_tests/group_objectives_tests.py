import contextlib
import sys
import typing as t

import numpy as np
import ray
from ray.util.placement_group import PlacementGroup

import dede as dd

sys.setrecursionlimit(10000)


def test_sum(n, pg):
    N, M = n, n
    x = dd.Variable((N, M), nonneg=True)
    resource_constraints = [x[i, :].sum() >= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= j for j in range(M)]

    objective = dd.Maximize(dd.sum(x))

    prob = dd.Problem(objective, resource_constraints, demand_constraints)
    prob._get_grouped_objectives(pg)
    # result_dede = prob.solve(ray_address="auto", solver=dd.ECOS, num_cpus=num_cpus)


def test_weighted(n, pg):
    N, M = n, n
    x = dd.Variable((N, M), nonneg=True)
    w = 9 * np.random.uniform(0, 1, (N, M)) + 1
    bn = 9 * np.random.uniform(0, 1, (N,)) + 1
    bm = 9 * np.random.uniform(0, 1, (M,)) + 1

    resource_constraints = [x[i, :].sum() >= bn[i] for i in range(N)]
    demand_constraints = [x[:, j].sum() >= bm[j] for j in range(M)]

    objective = dd.Minimize(dd.sum(dd.multiply(x, w)))

    prob = dd.Problem(objective, resource_constraints, demand_constraints)
    prob._get_grouped_objectives(pg)
    # result_dede = prob.solve(ray_address="auto", num_cpus=num_cpus, solver=dd.ECOS)


def test_log(n, pg):
    N, M = n, n
    x = dd.Variable((N, M), nonneg=True)
    resource_constraints = [x[i, :].sum() >= (i + 1) * M for i in range(N)]
    demand_constraints = [x[:, j].sum() <= (j + 1) * N for j in range(M)]

    objective = dd.Maximize(dd.sum([dd.log(dd.sum(x[i])) for i in range(N)]))

    prob = dd.Problem(objective, resource_constraints, demand_constraints)
    prob._get_grouped_objectives(pg)
    # result_dede = prob.solve(ray_address="auto", num_cpus=num_cpus, solver=dd.SCS)


@contextlib.contextmanager
def get_pg(**kwargs) -> t.Iterator[PlacementGroup]:
    """Returns a placement group that tries to spread
    workers across all available nodes in the ray network.

    Ideally used as a context manager to free up
    the placement group once execution has finished.

    Args:
        max_cpus_per_node (int): how many CPUs to request per node. This is an upper bound on the
        number of CPUs reserved.

    Returns:
        t.Iterator[PlacementGroup]: the placement group
    """
    bundles = [{"CPU": 1.0}] * kwargs.get("num_cpus", 1)
    pg = ray.util.placement_group(bundles, strategy=kwargs.get("strategy", "PACK"))
    ray.get(pg.ready())
    try:
        yield pg
    finally:
        ray.util.remove_placement_group(pg)


if __name__ == "__main__":
    sum_multiplier = 100
    weighted_multiplier = 30
    log_multiplier = 10

    kwargs = [{"num_cpus": num_cpus, "strategy": "PACK"} for num_cpus in [1, 2, 4, 8, 16]] + [
        {"num_cpus": num_cpus, "strategy": "SPREAD"} for num_cpus in [4, 8, 16, 32]
    ]

    for multiplier in range(11, 31):
        for kwarg in kwargs:
            sum_n = multiplier * sum_multiplier
            weighted_n = multiplier * weighted_multiplier
            log_n = multiplier * log_multiplier

            with get_pg(**kwarg) as pg:
                print(
                    f"Testing sum n={sum_n}, num_cpus={kwarg.get('num_cpus')}, strategy={kwarg.get('strategy')}"
                )
                try:
                    test_sum(sum_n, pg)
                except Exception as e:
                    print(f"Error in test_sum with n={sum_n}, {kwarg}: {e}")

                print(
                    f"Testing weighted n={weighted_n}, num_cpus={kwarg.get('num_cpus')}, strategy={kwarg.get('strategy')}"
                )
                try:
                    test_weighted(weighted_n, pg)
                except Exception as e:
                    print(f"Error in test_weighted with n={weighted_n}, {kwarg}: {e}")

                print(
                    f"Testing log n={log_n}, num_cpus={kwarg.get('num_cpus')}, strategy={kwarg.get('strategy')}"
                )
                try:
                    test_log(log_n, pg)
                except Exception as e:
                    print(f"Error in test_log with n={log_n}, {kwarg}: {e}")
