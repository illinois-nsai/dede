import typing as t

import cvxpy as cp
import numpy as np
import ray
from numpy.typing import NDArray

from dede.utils import VarInfoT

from .subproblem import Subproblem


class SubproblemsWrap:
    """Wrap subproblems for one actor in ray."""

    def __init__(
        self,
        idx_r: list[int],
        idx_d: list[int],
        obj_gps_r: t.Sequence[cp.Expression],
        obj_gps_d: t.Sequence[cp.Expression],
        constrs_gps_r: list[list[cp.Constraint]],
        constrs_gps_d: list[list[cp.Constraint]],
        var_id_to_pos_gps_r: list[list[list[VarInfoT]]],
        var_id_to_pos_gps_d: list[list[list[VarInfoT]]],
        var_id_pos_set_r: set[VarInfoT],
        var_id_pos_set_d: set[VarInfoT],
        rho: float,
    ):
        # sort subproblem for better data locality
        self.probs_r: list[Subproblem] = []
        i: int
        for i in np.argsort(idx_r):
            # build resource problems
            idx, constrs_gp = idx_r[i], constrs_gps_r[i]
            obj_r = obj_gps_r[i]
            var_id_to_pos_gp = var_id_to_pos_gps_r[i]
            self.probs_r.append(
                Subproblem((0, idx), obj_r, constrs_gp, var_id_to_pos_gp, var_id_pos_set_d, rho)
            )
        self.probs_d: list[Subproblem] = []
        for i in np.argsort(idx_d):
            # build demand problems
            idx, constrs_gp = idx_d[i], constrs_gps_d[i]
            obj_d = obj_gps_d[i]
            var_id_to_pos_gp = var_id_to_pos_gps_d[i]
            self.probs_d.append(
                Subproblem((1, idx), obj_d, constrs_gp, var_id_to_pos_gp, var_id_pos_set_r, rho)
            )

        # maintain the parameter copy in the current thread
        self.param_id_to_param: dict[int, cp.Parameter] = {}
        for constrs_gp in constrs_gps_r + constrs_gps_d:
            for constr in constrs_gp:
                for param in t.cast(list[cp.Parameter], constr.parameters()):
                    self.param_id_to_param[t.cast(int, param.id)] = param

    @ray.method
    def get_solution_idx_r(self) -> list[VarInfoT]:
        """Record how to split a long input for resources."""
        sol_idx_r: list[VarInfoT] = []
        self.sol_split_r: list[int] = []
        for prob in self.probs_r:
            sol_idx_r += prob.get_solution_idx()
            self.sol_split_r.append(len(sol_idx_r))
        self.sol_split_r = self.sol_split_r[:-1]
        return sol_idx_r

    @ray.method
    def get_solution_idx_d(self) -> list[VarInfoT]:
        """Record how to split a long input for demands."""
        sol_idx_d: list[VarInfoT] = []
        self.sol_split_d: list[int] = []
        for prob in self.probs_d:
            sol_idx_d += prob.get_solution_idx()
            self.sol_split_d.append(len(sol_idx_d))
        self.sol_split_d = self.sol_split_d[:-1]
        return sol_idx_d

    @ray.method
    def get_solution_r(self) -> NDArray[np.floating[t.Any]]:
        """Get concatenated solution of resource problems."""
        if self.probs_r:
            return np.hstack([prob.get_solution() for prob in self.probs_r])
        else:
            return np.array([])

    @ray.method
    def get_solution_d(self) -> NDArray[np.floating[t.Any]]:
        """Get concatenated solution of demand problems."""
        if self.probs_d:
            return np.hstack([prob.get_solution() for prob in self.probs_d])
        else:
            return np.array([])

    @ray.method
    def get_local_solution_idx(self) -> list[VarInfoT]:
        """Get (var_id, position) of all local-only variables."""
        probs = self.probs_r + self.probs_d
        return [idx for prob in probs for idx in prob.get_local_solution_idx()]

    @ray.method
    def get_local_solution(self) -> list[NDArray[np.floating[t.Any]]]:
        """Get concatenated solution of all local-only variables."""
        probs = self.probs_r + self.probs_d
        return [sol for prob in probs for sol in prob.get_local_solution()]

    @ray.method
    def update_parameters(self, param_id_to_value: dict[int, NDArray[np.floating[t.Any]]]) -> None:
        """Update parameter value in the current actor."""
        for param_id, param in self.param_id_to_param.items():
            if param_id in param_id_to_value:
                param.value = param_id_to_value[t.cast(int, param.id)]

    @ray.method
    def solve_r(
        self, param_values: NDArray[np.floating[t.Any]], *args, **kwargs
    ) -> np.floating[t.Any]:
        """Solve resource problems in the current actor sequentially."""
        param_value_list = np.split(param_values, self.sol_split_r)
        aug_lgr = 0
        for prob, param_value in zip(self.probs_r, param_value_list):
            aug_lgr += prob.solve(param_value, *args, **kwargs)
        return t.cast(np.floating[t.Any], aug_lgr)

    @ray.method
    def solve_d(
        self, param_values: NDArray[np.floating[t.Any]], *args, **kwargs
    ) -> np.floating[t.Any]:
        """Solve demand problems in the current actor sequentially."""
        param_value_list = np.split(param_values, self.sol_split_d)
        aug_lgr = 0
        for prob, param_value in zip(self.probs_d, param_value_list):
            aug_lgr += prob.solve(param_value, *args, **kwargs)

        return t.cast(np.floating[t.Any], aug_lgr)

    @ray.method
    def get_obj(self) -> np.floating[t.Any]:
        """Get the sum of objective values."""
        obj = 0
        for prob in self.probs_r + self.probs_d:
            obj += prob.get_obj()
        return t.cast(np.floating[t.Any], obj)
