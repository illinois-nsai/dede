import csv
import time
from pathlib import Path

import ray

import dede as dd

ray.init(address="auto")


def test(n):
    N, M = n, n
    x = dd.Variable((N, M), nonneg=True)
    resource_constraints = [x[i, :].sum() >= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= j for j in range(M)]

    objective = dd.Maximize(dd.sum(x))

    prob = dd.Problem(objective, resource_constraints, demand_constraints)
    start = time.time()
    result_dede = prob.solve(solver=dd.ECOS)
    time_dede = time.time() - start

    return (n, time_dede)


if __name__ == "__main__":
    arr = []
    for i in [10, 50, 100, 200]:
        print(i)
        res = test(i)
        print(res)
        arr.append(res)

    # Writing to the CSV
    with open(Path.home() / "output.csv", "w", newline="") as f:
        writer = csv.writer(f)

        # Optional: Write a header
        writer.writerow(["n", "num_cpus", "time"])

        # Write the entire list of tuples at once
        writer.writerows(arr)
