import csv
import time

import dede as dd


def test(n: int, num_cpus: int) -> tuple:
    N, M = n, n
    x = dd.Variable((N, M), nonneg=True)
    resource_constraints = [x[i, :].sum() >= (i + 1) * M for i in range(N)]
    demand_constraints = [x[:, j].sum() <= (j + 1) * N for j in range(M)]

    objective = dd.Maximize(dd.sum([dd.log(dd.sum(x[i])) for i in range(N)]))

    prob = dd.Problem(objective, resource_constraints, demand_constraints)
    start = time.time()
    result_dede = prob.solve(solver=dd.SCS, num_cpus=num_cpus)
    time_dede = time.time() - start

    return (n, num_cpus, time_dede)

    with open("timing.txt", "a") as f:
        f.write(f"{n} {time_dede} {result_dede}\n")


if __name__ == "__main__":
    arr = []
    for num_cpus in [1, 2, 4, 8, 16, 32]:
        for i in [10, 50, 100, 200]:
            print(i, num_cpus)
            res = test(i, num_cpus)
            print(res)
            arr.append(res)

    # Writing to the CSV
    with open("output.csv", "w", newline="") as f:
        writer = csv.writer(f)

        # Optional: Write a header
        writer.writerow(["n", "num_cpus", "time"])

        # Write the entire list of tuples at once
        writer.writerows(arr)
