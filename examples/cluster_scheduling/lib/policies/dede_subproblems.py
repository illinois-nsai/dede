import time

import cvxpy as cp
import numpy as np
import ray
from cvxpy.problems.problem import Problem as CpProblem

from .objective import Objective

EPS = 1e-6


@ray.remote
class SubproblemsWrap:
    """Wrap subproblems for one actor in ray."""

    def __init__(self, objective, idx_r, idx_d, M, N, num_workers, throughputs, scale_factors, rho):
        self._objective = objective
        self.M, self.N = M, N
        self.probs_r = []
        for i in range(len(idx_r)):
            idx, num_worker = idx_r[i], num_workers[i]
            self.probs_r.append(
                SubproblemR(
                    self._objective, (0, idx), self.M, self.N, num_worker, scale_factors, rho
                )
            )
        self.probs_d = []
        for i in range(len(idx_d)):
            idx, throughput = idx_d[i], throughputs[:, i]
            self.probs_d.append(SubproblemD(self._objective, (1, idx), self.M, throughput, rho))

    def get_solution_r(self):
        """Get concatenated solution of resource problems."""
        if self.probs_r:
            return np.vstack([prob.get_solution() for prob in self.probs_r])
        else:
            return np.empty((0, self.N))

    def get_solution_d(self):
        """Get concatenated solution of demand problems."""
        if self.probs_d:
            return np.vstack([prob.get_solution() for prob in self.probs_d])
        elif self._objective == Objective.TOTAL_UTIL:
            return np.empty((0, self.M))
        elif self._objective == Objective.MAX_MIN_ALLOC:
            return np.empty((0, self.M + 1))

    def solve_r(self, param_values, *args, **kwargs):
        """Solve resource problems in the current actor sequentially."""
        aug_lgr = 0
        for prob, param_value in zip(self.probs_r, param_values):
            aug_lgr += prob.solve(param_value, *args, **kwargs)
        return aug_lgr

    def solve_d(self, param_values, *args, **kwargs):
        """Solve demand problems in the current actor sequentially."""
        aug_lgr = 0
        for prob, param_value in zip(self.probs_d, param_values):
            aug_lgr += prob.solve(param_value, *args, **kwargs)
        return aug_lgr

    def get_obj(self):
        """Get the sum of objective values."""
        obj = [prob.get_obj() for prob in self.probs_d if prob.is_valid]
        return obj

    def fix_r(self, param_values=None, iter=None):
        """Fix constraints violation of resource problems."""
        if self.probs_r:
            return np.vstack(
                [
                    prob.fix(param_value, iter)
                    for prob, param_value in zip(self.probs_r, param_values)
                ]
            )
        else:
            return np.empty((0, self.N))

    def fix_d(self, param_values=None, iter=None):
        """Fix constraints violation of demand problems."""
        if self.probs_d:
            return np.vstack(
                [
                    prob.fix(param_value, iter)
                    for prob, param_value in zip(self.probs_d, param_values)
                ]
            )
        else:
            return np.empty((0, self.M))

    def get_fix_obj(self, param_values=None):
        """Get the sum of objective values."""
        if param_values is not None:
            obj = [
                prob.get_fix_obj(param_value)
                for prob, param_value in zip(self.probs_d, param_values)
                if prob.is_valid
            ]
        else:
            obj = [prob.get_fix_obj() for prob in self.probs_d if prob.is_valid]
        return obj

    def update_parameters(self, scale_factors, throughputs, is_valid_idx_d):
        """Update parameter value in the current actor."""
        for prob in self.probs_r:
            prob.scale_factors.value = scale_factors
        for prob, throughput, is_valid in zip(self.probs_d, throughputs.T, is_valid_idx_d):
            prob.throughput.value = throughput
            prob.is_valid = is_valid
            if not is_valid:
                prob.invalid()

    def get_d_t(self):
        t = []
        for prob in self.probs_d:
            t.append(prob.get_t())
        return t

    def get_r_t(self):
        t = []
        for prob in self.probs_r:
            t.append(prob.get_t())
        return t


class SubproblemR(CpProblem):
    def __init__(self, objective, idx, M, N, num_worker, scale_factors, rho):
        self._objective_name = objective
        self._runtime = 0
        self.id = idx
        self.M = M
        self.N = N
        self.rho = rho

        self.scale_factors = cp.Parameter(N, value=scale_factors)
        self.num_worker = num_worker

        self.var = cp.Variable(N, nonneg=True)
        self.var.value = np.zeros(self.var.shape)
        self.s = cp.Variable(nonneg=True)
        self.s.value = 0
        # prevent None solution in rare cases
        self.old_var_value = np.zeros(N)
        self.old_s_value = 0.0
        self.param = cp.Parameter(N, value=np.zeros(N))

        self.f1 = self.var @ self.scale_factors + self.s - self.num_worker
        self.l1 = cp.Parameter(value=0)
        self.f2 = self.var - self.param
        self.l2 = cp.Parameter(self.f2.shape, value=np.zeros(self.f2.shape))

        super(SubproblemR, self).__init__(
            cp.Minimize(
                cp.sum_squares(self.f1 + self.l1)
                + cp.sum_squares(self.var)
                - 2 * self.var @ self.param
                + 2 * self.var @ self.l2
            ),
            # cp.sum_squares(self.f2 + self.l2)),
        )

    def get_solution(self):
        return self.var.value

    def fix(self, param_value=None, iter=None):
        if self._objective_name == Objective.TOTAL_UTIL:
            if iter == 0:
                return (
                    param_value
                    / max(param_value @ self.scale_factors.value, EPS, self.num_worker)
                    * self.num_worker
                )
            else:
                return (
                    param_value / max(param_value @ self.scale_factors.value, EPS) * self.num_worker
                )
        elif self._objective_name == Objective.MAX_MIN_ALLOC:
            if iter >= 0:
                return (
                    param_value
                    / max(param_value @ self.scale_factors.value, EPS, self.num_worker)
                    * self.num_worker
                )
            else:
                return (
                    param_value / max(param_value @ self.scale_factors.value, EPS) * self.num_worker
                )

    def solve(self, param_value, *args, **kwargs):
        start = time.time()
        self.l1.value += self.f1.value
        self.param.value = param_value
        self.l2.value += self.f2.value
        self.old_var_value = self.var.value
        self.old_s_value = self.s.value
        res = super(SubproblemR, self).solve(*args, **kwargs)
        if self.var.value is None:
            self.var.value = self.old_var_value
            self.s.value = self.old_s_value
        self._runtime = time.time() - start
        return res

    def get_t(self):
        return [
            self._runtime,
            self.solver_stats.solve_time if self.solver_stats is not None else 0,
            self.compilation_time if self.compilation_time is not None else 0,
        ]


class SubproblemD(CpProblem):
    def __init__(self, objective, idx, M, throughput, rho):
        self._objective_name = objective
        self._runtime = 0
        self.is_valid = 0
        self.id = idx
        self.M = M
        self.rho = rho

        self.throughput = cp.Parameter(M, value=throughput)

        if self._objective_name == Objective.TOTAL_UTIL:
            self.var = cp.Variable(M, nonneg=True)
            self.var.value = np.zeros(M)
            self.s = cp.Variable(nonneg=True)
            self.s.value = 0
            self.old_var_value = np.zeros(M)
            self.old_s_value = 0.0
            self.param = cp.Parameter(M, value=np.zeros(M))

            self.f1 = self.var.sum() + self.s - 1
            self.l1 = cp.Parameter(value=0)
            self.f2 = self.var - self.param
            self.l2 = cp.Parameter(self.f2.shape, value=np.zeros(self.f2.shape))

            super(SubproblemD, self).__init__(
                cp.Minimize(
                    -cp.log(self.throughput @ self.var + EPS) * 2 / self.rho
                    + cp.sum_squares(self.f1 + self.l1)
                    + cp.sum_squares(self.var)
                    - 2 * self.var @ self.param
                    + 2 * self.var @ self.l2
                ),
                # cp.sum_squares(self.f2 + self.l2)),
            )

        elif self._objective_name == Objective.MAX_MIN_ALLOC:
            self.var = cp.Variable(M + 1, nonneg=True)
            self.var.value = np.zeros(M + 1)
            self.s = cp.Variable(2, nonneg=True)
            self.s.value = np.zeros(2)
            self.old_var_value = np.zeros(M + 1)
            self.old_s_value = np.zeros(2)
            self.param = cp.Parameter(M + 1, value=np.zeros(M + 1))

            self.f1 = cp.hstack(
                [
                    self.throughput @ self.var[:-1] - self.s[0] - self.var[-1],
                    self.var[:-1].sum() + self.s[1] - 1,
                ]
            )
            self.l1 = cp.Parameter(self.f1.shape, value=np.zeros(self.f1.shape))
            self.f2 = self.var - self.param
            self.l2 = cp.Parameter(self.f2.shape, value=np.zeros(self.f2.shape))

            super(SubproblemD, self).__init__(
                cp.Minimize(
                    cp.sum_squares(self.f1 + self.l1)
                    + cp.sum_squares(self.var)
                    - 2 * self.var @ self.param
                    + 2 * self.var @ self.l2
                ),
                # cp.sum_squares(self.f2 + self.l2)),
            )

    def invalid(self):
        self.var.value *= 0
        self.s.value *= 0
        self.param.value *= 0
        self.l1.value *= 0
        self.l2.value *= 0

    def get_solution(self):
        if not self.is_valid:
            return np.zeros(self.var.shape)
        else:
            return self.var.value

    def get_obj(self):
        if not self.is_valid:
            return 0
        if self._objective_name == Objective.TOTAL_UTIL:
            return -np.log(self.throughput.value @ self.var.value + EPS)
        elif self._objective_name == Objective.MAX_MIN_ALLOC:
            return self.var[-1].value

    def fix(self, param_value=None, iter=None):
        if not self.is_valid:
            return np.zeros(self.M)
        if self._objective_name == Objective.TOTAL_UTIL:
            if iter == 0:
                self.var_fix = param_value / max(param_value.sum(), 1)
            elif param_value.sum() - self.var_fix.sum() > EPS:
                delta = param_value - self.var_fix
                self.var_fix += (
                    delta / max(delta.sum(), 1 - self.var_fix.sum(), EPS) * (1 - self.var_fix.sum())
                )
            return self.var_fix
        elif self._objective_name == Objective.MAX_MIN_ALLOC:
            if iter >= 0:
                self.var_fix = param_value[:-1] / (param_value[:-1].sum() + EPS)
                self.var_fix = self.var_fix / max(
                    self.throughput.value @ self.var_fix / (param_value[-1] * 1.01 + EPS), 1
                )
            elif param_value.sum() - self.var_fix.sum() > EPS:
                delta = param_value - self.var_fix
                self.var_fix += (
                    delta / max(delta.sum(), 1 - self.var_fix.sum(), EPS) * (1 - self.var_fix.sum())
                )
            return self.var_fix

    def get_fix_obj(self, param_value=None):
        if not self.is_valid:
            return 0
        if self._objective_name == Objective.TOTAL_UTIL:
            return -np.log(self.throughput.value @ self.var_fix + EPS)
        elif self._objective_name == Objective.MAX_MIN_ALLOC:
            return self.throughput.value @ param_value

    def solve(self, param_value, *args, **kwargs):
        if not self.is_valid:
            return 0
        else:
            start = time.time()
            self.l1.value += self.f1.value
            # note: we haven't update param yet
            self.l2.value += self.f2.value
            self.param.value = param_value.copy()
            self.old_var_value = self.var.value
            self.old_s_value = self.s.value
            res = super(SubproblemD, self).solve(*args, **kwargs)
            if self.var.value is None:
                self.var.value = self.old_var_value
                self.s.value = self.old_s_value
            if self._objective_name == Objective.TOTAL_UTIL:
                self.var_fix = self.var.value
            elif self._objective_name == Objective.MAX_MIN_ALLOC:
                self.var_fix = self.var.value[:-1]
            self._runtime = time.time() - start
            return res

    def get_t(self):
        if not self.is_valid:
            return [0, 0, 0]
        else:
            return [self._runtime, self.solver_stats.solve_time, self.compilation_time]
