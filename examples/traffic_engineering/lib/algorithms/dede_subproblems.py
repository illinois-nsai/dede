import time

import cvxpy as cp
import numpy as np
import ray
from cvxpy.problems.problem import Problem as CpProblem
from scipy.sparse import coo_matrix

from .abstract_formulation import Objective

rtol = 1e-4
atol = 1e-4
EPS = 1e-6


@ray.remote
class SubproblemsWrap:
    """Wrap subproblems for one actor in ray."""

    def __init__(
        self, objective, num_nodes, num_edges, idx_r, idx_d, constrs_gps_r, constrs_gps_d, rho
    ):
        self._objective = objective
        self.num_nodes = num_nodes
        self.num_edges = num_edges
        self.probs_r = []
        for i in range(len(idx_r)):
            idx, constrs_gp = idx_r[i], constrs_gps_r[i]
            self.probs_r.append(
                SubproblemR(self._objective, num_nodes, num_edges, (0, idx), constrs_gp, rho)
            )
        self.probs_d = []
        for i in range(len(idx_d)):
            idx, constrs_gp = idx_d[i], constrs_gps_d[i]
            self.probs_d.append(
                SubproblemD(self._objective, num_nodes, num_edges, (1, idx), constrs_gp, rho)
            )

    def get_solution_r(self):
        """Get concatenated solution of resource problems."""
        if self.probs_r:
            return np.vstack([prob.get_solution() for prob in self.probs_r])
        elif self._objective == Objective.TOTAL_FLOW:
            return np.empty((0, self.num_nodes))
        elif self._objective == Objective.MIN_MAX_LINK_UTIL:
            return np.empty((0, self.num_nodes + 1))

    def get_solution_d(self):
        """Get concatenated solution of demand problems."""
        if self.probs_d:
            return np.array([prob.get_solution() for prob in self.probs_d])
        else:
            return np.empty((0, self.num_edges))

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
        """Get the sum of objective values. Constraints may be violated."""
        obj = 0
        for prob in self.probs_r + self.probs_d:
            if self._objective == Objective.TOTAL_FLOW:
                obj += prob.get_obj()
            elif self._objective == Objective.MIN_MAX_LINK_UTIL:
                obj = max(obj, prob.get_obj())
        return obj

    def fix_r(self, param_values=None):
        """Fix constraints violation of resource problems."""
        if self.probs_r:
            return np.vstack(
                [prob.fix(param_value) for prob, param_value in zip(self.probs_r, param_values)]
            )
        else:
            return np.empty((0, self.num_nodes))

    def fix_d(self, param_values=None):
        """Fix constraints violation of demand problems."""
        if self.probs_d:
            return np.vstack(
                [prob.fix(param_value) for prob, param_value in zip(self.probs_d, param_values)]
            )
        else:
            return np.empty((0, self.num_edges))

    def get_fix_obj(self, param_values=None):
        """Get the sum of objective values after fixing constraint violations."""
        obj = 0
        if self._objective == Objective.TOTAL_FLOW:
            for prob in self.probs_d:
                obj += prob.get_fix_obj()
        elif self._objective == Objective.MIN_MAX_LINK_UTIL:
            for prob, param_value in zip(self.probs_r, param_values):
                obj = max(obj, prob.get_fix_obj(param_value))
        return obj

    def update_parameters(self, param_values):
        """Update parameter value in the current actor."""
        for prob, param_value in zip(self.probs_d, param_values):
            prob.demands.value = param_value
            # not update lambda for demands w/o large changes
            # prob.l1.value *= 0
            # prob.l2.value *= 0
            if self._objective == Objective.MIN_MAX_LINK_UTIL:
                prob.demands_on_edge.value = prob.m_all_e_all_d @ param_value

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
    def __init__(self, objective, num_nodes, num_edges, idx, constrs_gp, rho):
        self._objective_name = objective
        self._runtime = 0
        self.id = idx
        self.rho = rho
        self.capacity = constrs_gp

        if self._objective_name == Objective.TOTAL_FLOW:
            self.var = cp.Variable(num_nodes, nonneg=True)
            self.var.value = np.zeros(num_nodes)
            self.s = cp.Variable(nonneg=True)
            self.s.value = 0.0
            # prevent None solution in rare cases
            self.old_var_value = np.zeros(num_nodes)
            self.old_s_value = 0.0
            self.param = cp.Parameter(num_nodes)
            self.param.value = np.zeros(num_nodes)

            # lambda for original constraints
            self.f1 = self.var.sum() + self.s - self.capacity
            self.l1 = cp.Parameter(value=0.0)
            # lambda for x = z
            self.f2 = self.var - self.param
            self.l2 = cp.Parameter(self.f2.shape, value=np.zeros(self.f2.shape))

            super(SubproblemR, self).__init__(
                cp.Minimize(
                    self.rho / 2 * cp.sum_squares(self.f1 + self.l1)
                    + self.rho / 2 * cp.sum_squares(self.var)
                    - self.rho * self.var @ self.param
                    + self.rho * self.var @ self.l2
                )
                # self.rho / 2 * cp.sum_squares(self.f2 + self.l2))
            )
        elif self._objective_name == Objective.MIN_MAX_LINK_UTIL:
            self.var = cp.Variable(num_nodes + 1, nonneg=True)
            self.var.value = np.zeros(num_nodes + 1)
            self.s = cp.Variable(nonneg=True)
            self.s.value = 0.0
            # prevent None solution in rare cases
            self.old_var_value = np.zeros(num_nodes + 1)
            self.old_s_value = 0.0
            self.param = cp.Parameter(num_nodes + 1)
            self.param.value = np.zeros(num_nodes + 1)

            # lambda for original constraints
            self.f1 = self.var[:-1].sum() + self.s - self.capacity * self.var[-1]
            self.l1 = cp.Parameter(value=0.0)
            # lambda for x = z
            self.f2 = self.var - self.param
            self.l2 = cp.Parameter(self.f2.shape, value=np.zeros(self.f2.shape))

            super(SubproblemR, self).__init__(
                cp.Minimize(
                    cp.sum_squares(self.f1 + self.l1)
                    + cp.sum_squares(self.var)
                    - 2 * self.var @ self.param
                    + 2 * self.var @ self.l2
                )
                # cp.sum_squares(self.f2 + self.l2))
            )

    def get_solution(self):
        return self.var.value

    def get_obj(self):
        if self._objective_name == Objective.TOTAL_FLOW:
            return 0
        elif self._objective_name == Objective.MIN_MAX_LINK_UTIL:
            return self.var.value[-1]

    def solve(self, param_value, *args, **kwargs):
        start = time.time()
        self.l1.value += self.f1.value
        self.param.value = param_value
        self.l2.value += self.f2.value
        self.old_var_value = self.var.value
        self.old_s_value = self.s.value
        res = super(SubproblemR, self).solve(*args, **kwargs)
        # prevent None solution in rare cases
        if self.var.value is None:
            self.var.value = self.old_var_value
            self.s.value = self.old_s_value
        self._runtime = time.time() - start
        return res

    def fix(self, param_value=None):
        if self._objective_name == Objective.TOTAL_FLOW:
            return param_value / (param_value.sum() + EPS) * self.capacity
        elif self._objective_name == Objective.MIN_MAX_LINK_UTIL:
            return param_value[:-1] / (param_value.sum() + EPS) * self.capacity * param_value[-1]

    def get_fix_obj(self, param_value=None):
        if self._objective_name == Objective.TOTAL_FLOW:
            return 0
        elif self._objective_name == Objective.MIN_MAX_LINK_UTIL:
            return param_value.sum() / (self.capacity + EPS)

    def get_t(self):
        return [self._runtime, self.solver_stats.solve_time, self.compilation_time]


class SubproblemD(CpProblem):
    def __init__(self, objective, num_nodes, num_edges, idx, constrs_gp, rho):
        self._objective_name = objective
        self._runtime = 0
        self.id = idx
        self.rho = rho

        self.num_paths = 0
        self.m_all_d_all_p, self.m_all_e_all_p = [], []
        for dst, demand, idx_paths in zip(*constrs_gp):
            for idx_p in idx_paths:
                self.m_all_d_all_p.append([dst, self.num_paths])
                self.m_all_e_all_p += [[idx_e, self.num_paths] for idx_e in idx_p]
                self.num_paths += 1
        e_to_shorter_e_dict = {}
        self.num_edges = num_edges
        # only focus on edge used
        self.idx_e_list = []
        for idx_e in np.sort(np.unique(np.array(self.m_all_e_all_p)[:, 0])):
            e_to_shorter_e_dict[idx_e] = len(e_to_shorter_e_dict)
            self.idx_e_list.append(idx_e)
        self.m_all_d_all_p = coo_matrix(
            (np.ones(len(self.m_all_d_all_p)), np.array(self.m_all_d_all_p).T),
            shape=(num_nodes, self.num_paths),
        )
        self.m_all_e_all_p = [[e_to_shorter_e_dict[idx_e], p] for idx_e, p in self.m_all_e_all_p]
        self.m_all_e_all_p = coo_matrix(
            (np.ones(len(self.m_all_e_all_p)), np.array(self.m_all_e_all_p).T),
            shape=(len(e_to_shorter_e_dict), self.num_paths),
        )

        if self._objective_name == Objective.TOTAL_FLOW:
            self.var = cp.Variable(self.num_paths, nonneg=True)
            self.var.value = np.zeros(self.num_paths)
            self.s = cp.Variable(num_nodes, nonneg=True)
            self.s.value = np.zeros(num_nodes)
            self.demands = cp.Parameter(num_nodes)
            self.demands.value = np.array(constrs_gp[1])
            self.param = cp.Parameter(len(e_to_shorter_e_dict))
            self.param.value = np.zeros(len(e_to_shorter_e_dict))

            # lambda for original constraints
            self.f1 = self.m_all_d_all_p @ self.var + self.s - self.demands
            self.l1 = cp.Parameter(self.f1.shape, value=np.zeros(self.f1.shape))
            # lambda for x = z
            self.f2 = self.m_all_e_all_p @ self.var - self.param
            self.l2 = cp.Parameter(self.f2.shape, value=np.zeros(self.f2.shape))

            super(SubproblemD, self).__init__(
                cp.Minimize(
                    -self.var.sum()
                    + self.rho / 2 * cp.sum_squares(self.f1 + self.l1)
                    + self.rho / 2 * cp.sum_squares(self.f2 + self.l2)
                ),
            )

        elif self._objective_name == Objective.MIN_MAX_LINK_UTIL:
            self.num_rest_paths, self.num_paths = 0, 0
            self.m_all_d_rest_p, self.m_all_e_rest_p, self.m_all_e_all_d = [], [], []
            self.rest_p_all_p_list, self.first_p_all_p_list = [], []
            for dst, demand, idx_paths in zip(*constrs_gp):
                idx_e_dict = {}
                if not idx_paths:
                    continue
                for idx_e in idx_paths[0]:
                    self.m_all_e_all_d.append([idx_e, dst])
                    idx_e_dict[idx_e] = np.arange(
                        self.num_rest_paths, self.num_rest_paths + len(idx_paths) - 1
                    ).tolist()
                self.first_p_all_p_list.append(self.num_paths)
                self.num_paths += 1
                for idx_p in idx_paths[1:]:
                    # the last path allocation is demands minus allocation sum expect the last path
                    self.m_all_d_rest_p.append([dst, self.num_rest_paths])
                    self.rest_p_all_p_list.append(self.num_paths)
                    for idx_e in idx_p:
                        if idx_e not in idx_e_dict:
                            self.m_all_e_rest_p.append([idx_e, self.num_rest_paths, 1])
                        else:
                            idx_e_dict[idx_e].remove(self.num_rest_paths)
                    self.num_rest_paths += 1
                    self.num_paths += 1
                for idx_e, ps in idx_e_dict.items():
                    for p in ps:
                        self.m_all_e_rest_p.append([idx_e, p, -1])

            self.m_all_e_all_d = [
                [e_to_shorter_e_dict[idx_e], d] for idx_e, d in self.m_all_e_all_d
            ]
            self.m_all_e_all_d = coo_matrix(
                (np.ones(len(self.m_all_e_all_d)), np.array(self.m_all_e_all_d).T),
                shape=(len(e_to_shorter_e_dict), num_nodes),
            )
            self.demands = cp.Parameter(num_nodes)
            self.demands.value = np.array(constrs_gp[1])
            self.demands_on_edge = cp.Parameter(len(e_to_shorter_e_dict))
            self.demands_on_edge.value = self.m_all_e_all_d @ self.demands.value

            if self.num_rest_paths > 0:
                self.m_all_d_rest_p = coo_matrix(
                    (np.ones(len(self.m_all_d_rest_p)), np.array(self.m_all_d_rest_p).T),
                    shape=(num_nodes, self.num_rest_paths),
                )
                self.m_all_e_rest_p = np.array(
                    [[e_to_shorter_e_dict[idx_e], p, val] for idx_e, p, val in self.m_all_e_rest_p]
                ).T
                self.m_all_e_rest_p = coo_matrix(
                    (self.m_all_e_rest_p[-1], self.m_all_e_rest_p[:-1]),
                    shape=(len(e_to_shorter_e_dict), self.num_rest_paths),
                )

                self.param = cp.Parameter(len(e_to_shorter_e_dict))
                self.param.value = np.zeros(len(e_to_shorter_e_dict))

                self.var = cp.Variable(self.num_rest_paths, nonneg=True)
                self.var.value = np.zeros(self.num_rest_paths)
                self.s = cp.Variable(num_nodes, nonneg=True)
                self.s.value = np.zeros(num_nodes)

                # lambda for original constraints
                self.f1 = self.m_all_d_rest_p @ self.var + self.s - self.demands
                self.l1 = cp.Parameter(self.f1.shape, value=np.zeros(self.f1.shape))
                # lambda for x = z
                self.f2 = self.m_all_e_rest_p @ self.var + self.demands_on_edge - self.param
                self.l2 = cp.Parameter(self.f2.shape, value=np.zeros(self.f2.shape))

                super(SubproblemD, self).__init__(
                    cp.Minimize(
                        cp.sum_squares(self.f1 + self.l1) + cp.sum_squares(self.f2 + self.l2)
                    ),
                )
            else:
                super(SubproblemD, self).__init__(cp.Minimize(0))

    def get_solution(self):
        res = np.zeros(self.num_edges)
        if self._objective_name == Objective.TOTAL_FLOW:
            res[self.idx_e_list] = self.m_all_e_all_p @ self.var.value
        elif self._objective_name == Objective.MIN_MAX_LINK_UTIL:
            if self.num_rest_paths > 0:
                res[self.idx_e_list] = np.maximum(
                    self.m_all_e_rest_p @ self.var.value + self.demands_on_edge.value, 0
                )
            else:
                res[self.idx_e_list] = self.demands_on_edge.value
        return res

    def get_obj(self):
        if self._objective_name == Objective.TOTAL_FLOW:
            return -self.var.value.sum()
        elif self._objective_name == Objective.MIN_MAX_LINK_UTIL:
            return 0

    def solve(self, param_value, *args, **kwargs):
        if self._objective_name == Objective.MIN_MAX_LINK_UTIL and self.num_rest_paths == 0:
            return super(SubproblemD, self).solve(*args, **kwargs)
        start = time.time()
        self.l1.value += self.f1.value
        # note: we haven't update param in f2 yet
        # note: keep l2 the same between resource probs and demand probs at each iter
        self.l2.value += self.f2.value
        self.param.value = param_value[self.idx_e_list]
        res = super(SubproblemD, self).solve(*args, **kwargs)
        self.var_fix = self.var.value
        self._runtime = time.time() - start
        return res

    def fix(self, param_value=None):
        res = np.zeros(self.num_edges)
        if self._objective_name == Objective.TOTAL_FLOW:
            # remove demand over-allocation
            self.var_fix = self.var.value / (
                np.maximum(self.m_all_d_all_p @ self.var.value / (self.demands.value + EPS), 1)
                @ self.m_all_d_all_p
            )
            # remove edge capcity over-allocation
            edge_oversubscribe = np.maximum(
                self.m_all_e_all_p @ self.var_fix / (param_value[self.idx_e_list] + EPS), 1
            )
            self.var_fix /= (
                coo_matrix(
                    (
                        [edge_oversubscribe[i] for i in self.m_all_e_all_p.row],
                        (self.m_all_e_all_p.row, self.m_all_e_all_p.col),
                    ),
                    shape=self.m_all_e_all_p.shape,
                )
                .max(axis=0)
                .toarray()
                .reshape(-1)
            )
            # no violation
            # np.testing.assert_array_less(self.m_all_d_all_p @ self.var_fix, self.demands.value * (1 + rtol) + atol)
            # np.testing.assert_array_less(self.m_all_e_all_p @ self.var_fix, param_value[self.idx_e_list] * (1 + rtol) + atol)
            res[self.idx_e_list] = self.m_all_e_all_p @ self.var_fix
        elif self._objective_name == Objective.MIN_MAX_LINK_UTIL:
            # allocate all demand
            if self.num_rest_paths > 0:
                self.all_var_fix = np.zeros(self.num_paths)
                self.all_var_fix[self.rest_p_all_p_list] = self.var_fix
                self.all_var_fix[self.first_p_all_p_list] = np.maximum(
                    np.delete(self.demands.value - self.m_all_d_rest_p @ self.var_fix, self.id[1]),
                    0,
                )
                # assume all the edge is at max utilization
                edge_oversubscribe = (
                    self.m_all_e_all_p @ self.all_var_fix / (param_value[self.idx_e_list] + EPS)
                )
                self.all_var_fix /= (
                    coo_matrix(
                        (
                            [edge_oversubscribe[i] for i in self.m_all_e_all_p.row],
                            (self.m_all_e_all_p.row, self.m_all_e_all_p.col),
                        ),
                        shape=self.m_all_e_all_p.shape,
                    )
                    .max(axis=0)
                    .toarray()
                    .reshape(-1)
                    + EPS
                )
                # remove demand over-allocation
                self.var_fix = self.all_var_fix[self.rest_p_all_p_list] / (
                    self.m_all_d_all_p
                    @ self.all_var_fix
                    / (self.demands.value + EPS)
                    @ self.m_all_d_rest_p
                    + EPS
                )
                res[self.idx_e_list] = (
                    self.m_all_e_rest_p @ self.var_fix + self.demands_on_edge.value
                )
                # no violation
                # np.testing.assert_array_less(self.m_all_d_rest_p @ self.var_fix, self.demands.value * (1 + rtol) + atol)

                # if we don't do iteration
                # self.var_fix = self.var.value / (np.maximum(self.m_all_d_rest_p @ self.var.value / (self.demands.value + EPS), 1) @ self.m_all_d_rest_p)
                # res[self.idx_e_list] = self.m_all_e_rest_p @ self.var_fix + self.demands_on_edge.value
            else:
                res[self.idx_e_list] = self.demands_on_edge.value
        return res

    def get_fix_obj(self, param_value=None):
        if self._objective_name == Objective.TOTAL_FLOW:
            return -self.var_fix.sum()
        elif self._objective_name == Objective.MIN_MAX_LINK_UTIL:
            return 0

    def get_t(self):
        return [
            self._runtime,
            self.solver_stats.solve_time if self.solver_stats.solve_time is not None else 0,
            self.compilation_time,
        ]
