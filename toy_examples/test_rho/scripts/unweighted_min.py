#!/usr/bin/env python3

import dede as dd
import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt


def test(N, M, rho_vals, num_iter=20):
    '''
    Solve with DeDe for multiple rhos, and with CVXPY.
    Return:
      - dict of rho -> DeDe result
      - cvxpy result
    '''
    # Create allocation variables
    x = dd.Variable((N, M), nonneg=True)

    # Create parameters
    param = dd.Parameter(N, value=list(np.arange(1, N + 1)))

    # Constraints
    resource_constraints = [x[i, :].sum() >= param[i] for i in range(N)]    # each resource must be used at least param[i] (to be worthwhile) 
    demand_constraints = [x[:, j].sum() >= 1 for j in range(M)]             # each job must have at 1 unit total of resources to run

    # Objective
    objective = dd.Minimize(dd.sum(x))                                      # minimize total resources used

    # Construct problem
    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    # Solve with DeDe for each rho
    dede_results = {}
    for rho_val in rho_vals:
        result_dede = prob.solve(num_cpus=4, rho=rho_val, solver=dd.ECOS, num_iter=num_iter)
        dede_results[rho_val] = result_dede

    # Solve with CVXPY
    prob_cvxpy = cp.Problem(objective, resource_constraints + demand_constraints)
    result_cvxpy = prob_cvxpy.solve()

    return dede_results, result_cvxpy


if __name__ == '__main__':
    rho_vals = [5, 10, 20, 50, 100]
    Ns = list(range(5, 101, 5))  # test problem sizes from 1 to 20 for speed

    # Store results: {rho: [results for each N]}, cvxpy_results[N]
    dede_results_all = {rho: [] for rho in rho_vals}
    cvxpy_results_all = []

    for N in Ns:
        print(f"Solving problem size N={N}")
        dede_res, cvx_res = test(N, N, rho_vals, num_iter=20)

        for rho in rho_vals:
            dede_results_all[rho].append(dede_res[rho])
        cvxpy_results_all.append(cvx_res)

    # Plotting
    plt.figure(figsize=(12, 7))

    for rho in rho_vals:
        rel_errors = []
        for dede_val, cvx_val in zip(dede_results_all[rho], cvxpy_results_all):
            rel_error = abs(dede_val - cvx_val) / abs(cvx_val)
            rel_errors.append(rel_error)
        plt.plot(Ns, rel_errors, label=f'DeDe rho={rho}', marker='o')

    plt.xlabel('Problem size N (N x N)')
    plt.ylabel('Relative error vs CVXPY')
    plt.title('Relative Error of DeDe vs CVXPY Objective for Different Rho values')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.savefig("graphs/unweighted_min.png")
