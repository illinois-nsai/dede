import math
import typing as t

import cvxpy as cp
import numpy as np

GUROBI_OPTS = {"Threads": 1}


def check_solution(
    dede_val: np.floating[t.Any],
    cvxpy_val: np.floating[t.Any],
    objective: cp.Objective,
    constraints: t.Optional[list[cp.Constraint]] = None,
    feas_tol: float = 1.0,
) -> bool:
    """Returns True if the DeDe solution is acceptable.

    Returns False if:
    - dede_val is not within 5% of cvxpy_val
    - dede_val is more than 5% away from cvxpy_val in the
        direction opposite to the optimization target
    - any constraint is violated beyond feas_tol (if constraint is passed in)
    """
    is_maximize = isinstance(objective, cp.Maximize)

    if constraints is not None:
        for constr in constraints:
            if np.any(constr.violation() > feas_tol):
                print("constraints not satisfied (outside threshold)", constr.violation())
                return False

    within_tol = math.isclose(dede_val, cvxpy_val, rel_tol=0.05, abs_tol=0.01)

    if within_tol:
        return True

    if is_maximize:
        if dede_val >= cvxpy_val:
            return True
    else:
        if dede_val <= cvxpy_val:
            return True

    print("solution not within tolerance and in the wrong direction", dede_val, cvxpy_val)

    return False
