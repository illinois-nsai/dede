import dede as dd
import math


def test_dede():
    N, M = 100, 100

    # Create allocation variables.
    x = dd.Variable((N, M), nonneg=True)

    # Create the constraints.
    resource_constraints = [x[i, :].sum() >= i for i in range(N)]
    demand_constraints = [x[:, j].sum() <= j for j in range(M)]

    # Create an objective.
    objective = dd.Minimize(x.sum())

    # Construct the problem.
    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    # Solve the problem with DeDe on a 4-core CPU.
    result_dede = prob.solve(num_cpus=4, solver=dd.ECOS)

    # Solve the problem with cvxpy
    result_cvxpy = prob.solve(enable_dede=False)

    # Compare
    assert math.isclose(result_dede, result_cvxpy, rel_tol=0.1, abs_tol=0.1)


def test_dede_with_param():
    N, M = 4, 4

    # Create allocation variables.
    x = dd.Variable((N, M), nonneg=True)

    # Create parameters.
    param = dd.Parameter(N, value=[1, 3, 5, 7])

    # Create the constraints.
    resource_constraints = [x[i].sum() >= param[i] for i in range(N)]
    demand_constraints = [x[:, j].sum() >= 1 for j in range(M)]

    # Create an objective.
    objective = dd.Minimize(x.sum())

    # Construct the problem.
    prob = dd.Problem(objective, resource_constraints, demand_constraints)

    # Solve the problem with DeDe on a 4-core CPU.
    result_dede = prob.solve(num_cpus=4, solver=dd.ECOS)

    # Solve the problem with cvxpy
    result_cvxpy = prob.solve(enable_dede=False)

    # Compare
    math.isclose(result_dede, result_cvxpy, rel_tol=0.1, abs_tol=0.1)

    # Change parameter value and re-solve
    param.value += 1

    # Solve the problem with DeDe on a 4-core CPU.
    result_dede = prob.solve(num_cpus=4, solver=dd.ECOS)

    # Solve the problem with cvxpy
    result_cvxpy = prob.solve(enable_dede=False)

    # Compare
    math.isclose(result_dede, result_cvxpy, rel_tol=0.1, abs_tol=0.1)
