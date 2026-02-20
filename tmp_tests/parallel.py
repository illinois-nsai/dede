import time

import dede as dd


def test(n):
    N, M = n, n
    x = dd.Variable((N, M), nonneg=True)
    resource_constraints = [x[i, :].sum() >= (i + 1) * M for i in range(N)]
    demand_constraints = [x[:, j].sum() <= (j + 1) * N for j in range(M)]

    objective = dd.Maximize(dd.sum([dd.log(dd.sum(x[i])) for i in range(N)]))

    prob = dd.Problem(objective, resource_constraints, demand_constraints)
    start = time.time()
    result_dede = prob.solve(solver=dd.SCS)
    time_dede = time.time() - start

    with open("timing.txt", "a") as f:
        f.write(f"{n} {time_dede} {result_dede}\n")


if __name__ == "__main__":
    for i in [100]:
        test(i)
