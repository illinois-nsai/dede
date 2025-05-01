import cvxpy as cp
import numpy as np
import time
import ray

from .subproblem import Subproblem


@ray.remote
class SubproblemsWrap():
    '''Wrap subproblems for one actor in ray.'''

    def __init__(
            self, idx_r, idx_d, obj_gps_r, obj_gps_d,
            constrs_gps_r, constrs_gps_d,
            var_id_to_pos_gps_r, var_id_to_pos_gps_d,
            var_id_pos_set_r, var_id_pos_set_d, rho):
        # sort subproblem for better data locality
        self.probs_r = []
        for i in np.argsort(idx_r):
            # build resource problems
            idx, constrs_gp = idx_r[i], constrs_gps_r[i]
            obj_r = obj_gps_r[i]
            var_id_to_pos_gp = var_id_to_pos_gps_r[i]
            self.probs_r.append(Subproblem(
                (0, idx),
                obj_r,
                constrs_gp,
                var_id_to_pos_gp,
                var_id_pos_set_d,
                rho))
        self.probs_d = []
        for i in np.argsort(idx_d):
            # build demand problems
            idx, constrs_gp = idx_d[i], constrs_gps_d[i]
            obj_d = obj_gps_d[i]
            var_id_to_pos_gp = var_id_to_pos_gps_d[i]
            self.probs_d.append(Subproblem(
                (1, idx),
                obj_d,
                constrs_gp,
                var_id_to_pos_gp,
                var_id_pos_set_r,
                rho))

        # maintain the parameter copy in the current thread
        self.param_id_to_param = {}
        for constrs_gp in constrs_gps_r + constrs_gps_d:
            for constr in constrs_gp:
                for param in constr.parameters():
                    self.param_id_to_param[param.id] = param

    def get_solution_idx_r(self):
        '''Record how to split a long input for resources.'''
        sol_idx_r = []
        self.sol_split_r = []
        for prob in self.probs_r:
            sol_idx_r += prob.get_solution_idx()
            self.sol_split_r.append(len(sol_idx_r))
        self.sol_split_r = self.sol_split_r[:-1]
        return sol_idx_r

    def get_solution_idx_d(self):
        '''Record how to split a long input for demands.'''
        sol_idx_d = []
        self.sol_split_d = []
        for prob in self.probs_d:
            sol_idx_d += prob.get_solution_idx()
            self.sol_split_d.append(len(sol_idx_d))
        self.sol_split_d = self.sol_split_d[:-1]
        return sol_idx_d

    def get_solution_r(self):
        '''Get concatenated solution of resource problems.'''
        if self.probs_r:
            return np.hstack([prob.get_solution() for prob in self.probs_r])
        else:
            return np.array([])

    def get_solution_d(self):
        '''Get concatenated solution of demand problems.'''
        if self.probs_d:
            return np.hstack([prob.get_solution() for prob in self.probs_d])
        else:
            return np.array([])

    def update_parameters(self, param_id_to_value):
        '''Update parameter value in the current actor.'''
        for param_id, param in self.param_id_to_param.items():
            if param_id in param_id_to_value:
                param.value = param_id_to_value[param.id]

    def solve_r(self, param_values, *args, **kwargs):
        '''Solve resource problems in the current actor sequentially.'''
        param_value_list = np.split(param_values, self.sol_split_r)
        aug_lgr = 0
        for prob, param_value in zip(self.probs_r, param_value_list):
            aug_lgr += prob.solve(param_value, *args, **kwargs)
        return aug_lgr

    def solve_d(self, param_values, *args, **kwargs):
        '''Solve demand problems in the current actor sequentially.'''
        param_value_list = np.split(param_values, self.sol_split_d)
        aug_lgr = 0
        for prob, param_value in zip(self.probs_d, param_value_list):
            aug_lgr += prob.solve(param_value, *args, **kwargs)
        return aug_lgr

    def get_obj(self):
        '''Get the sum of objective values.'''
        obj = 0
        for prob in self.probs_r + self.probs_d:
            obj += prob.get_obj()
        return obj
