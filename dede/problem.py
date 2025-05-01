import numpy as np
import cvxpy as cp
import time
import ray
import os
from collections import defaultdict

from cvxpy.problems.problem import Problem as CpProblem
from cvxpy.constraints.zero import Zero, Equality
from cvxpy.constraints.nonpos import Inequality
from cvxpy.expressions.variable import Variable
from cvxpy.problems.objective import Maximize, Minimize

from .utils import (
    expand_expr,
    get_var_id_pos_list_from_cone,
    get_var_id_pos_list_from_linear)
from .subproblems_wrap import SubproblemsWrap


class SubprobCache:
    '''Cache subproblems.'''

    def __init__(self):
        self.key = None
        self.rho = None
        self.num_cpus = None
        self.probs = None
        self.param_idx_r, self.param_idx_d = [], []

    def invalidate(self):
        self.key = None
        self.rho = None
        self.num_cpus = None
        self.probs = None
        self.param_idx_r, self.param_idx_d = [], []

    def make_key(self, rho, num_cpus):
        return (rho, num_cpus)


class Problem(CpProblem):
    '''Build a resource allocation problem.'''

    def __init__(self, objective, resource_constraints, demand_constraints):
        '''Initialize problem with the objective and constraints.
        Args:
            objective: Minimize or Maximize. The problem's objective
            resource_variables: list of resource constraints
            demand_variables: list of demand constraints
        '''
        self._constrs_r = [
            self.convert_inequality(constr) for constr in resource_constraints]
        self._constrs_d = [
            self.convert_inequality(constr) for constr in demand_constraints]
        self._subprob_cache = SubprobCache()

        # Initialize original problem
        super(Problem, self).__init__(
            objective if type(objective) == Minimize else Minimize(
                -objective.expr),
            self._constrs_r + self._constrs_d)

        # get a dict mapping from param_id to value
        self.param_id_to_param = {
            param.id: param for param in self.parameters()}

        # get a dict mapping from constraints to list of (var_id, position)
        self.constr_dict_r = self.get_constr_dict(self._constrs_r)
        self.constr_dict_d = self.get_constr_dict(self._constrs_d)

        # get constraints groups
        self.constrs_gps_r = self.group_constrs(
            self._constrs_r, self.constr_dict_r)
        self.constrs_gps_d = self.group_constrs(
            self._constrs_d, self.constr_dict_d)

        # get objective groups
        self._obj_expr_r, self._obj_expr_d = self.group_objective()

    def convert_inequality(self, constr):
        if isinstance(constr, Zero) or isinstance(constr, Equality):
            return constr
        elif isinstance(constr, Inequality):
            return constr.expr + cp.Variable(constr.shape, nonneg=True) == 0
        else:
            raise ValueError(
                f'Constraint {constr} is neither equality nor inequality.')

    def solve(
            self, enable_dede=True, num_cpus=None, rho=None, num_iter=None,
            *args, **kwargs):
        '''Compiles and solves the original problem.
        Args:
            enable_dede: whether to decouple and decompose with DeDe
            num_cpus: number of CPUs to use; all the CPUs available if None
            rho: rho value in ADMM; 1 if None
            num_iter: ADMM iterations; stop under < 1% improvement if None
        '''
        # solve the original problem
        if not enable_dede:
            start = time.time()
            super(Problem, self).solve(*args, **kwargs)
            end = time.time()
            self._total_time = end - start
            return self.value

        # initialize num_cpus, rho
        if num_cpus is None:
            if self._subprob_cache.num_cpus is None:
                num_cpus = os.cpu_count()
            else:
                num_cpus = self._subprob_cache.num_cpus
        if rho is None:
            if self._subprob_cache.rho is None:
                rho = 1
            else:
                rho = self._subprob_cache.rho
        # check whether num_cpus is more than all available
        if num_cpus > os.cpu_count():
            raise ValueError(
                f'{num_cpus} CPUs exceeds upper limit of {os.cpu_count()}.')

        # check whether settings has been changed
        key = self._subprob_cache.make_key(rho, num_cpus)
        if key != self._subprob_cache.key:
            # invalidate old settings
            self._subprob_cache.invalidate()
            self._subprob_cache.key = key
            self._subprob_cache.rho = rho
            # initialize ray
            ray.shutdown()
            self._subprob_cache.num_cpus = num_cpus
            ray.init(num_cpus=num_cpus)
            # store subproblem in last solution
            self._subprob_cache.probs = self.get_subproblems(num_cpus, rho)
            # store parameter index in z solutions for x problems
            self._subprob_cache.param_idx_r, \
                self._subprob_cache.param_idx_d = self.get_param_idx()
            # get demand solution
            self.sol_d = np.hstack(ray.get([prob.get_solution_d.remote(
            ) for prob in self._subprob_cache.probs]))

        # update parameter values
        param_id_to_value = {
            param_id: param.value
            for param_id, param in self.param_id_to_param.items()}
        ray.get([prob.update_parameters.remote(
            param_id_to_value) for prob in self._subprob_cache.probs])

        # solve problem
        # use num_iter if specifed
        # otherwise, stop under < 1% improvement or reach 10000 upper limit
        i, aug_lgr, aug_lgr_old = 0, 1, 2
        while (num_iter is not None and i < num_iter) or \
            (num_iter is None and i < 10000 and (
                i < 2 or abs((aug_lgr - aug_lgr_old)/aug_lgr_old) > 0.01)):

            # initialize start time, iteration, augmented Lagrangian
            start, i, aug_lgr_old, aug_lgr = time.time(), i + 1, aug_lgr, 0

            # resource allocation
            aug_lgr += sum(ray.get([
                prob.solve_r.remote(
                    self.sol_d[param_idx], *args, **kwargs
                ) for prob, param_idx in zip(
                    self._subprob_cache.probs,
                    self._subprob_cache.param_idx_r)]))
            self.sol_r = np.hstack(ray.get([prob.get_solution_r.remote(
            ) for prob in self._subprob_cache.probs]))

            # demand allocation
            aug_lgr += sum(ray.get([
                prob.solve_d.remote(
                    self.sol_r[param_idx], *args, **kwargs
                ) for prob, param_idx in zip(
                    self._subprob_cache.probs,
                    self._subprob_cache.param_idx_d)]))
            self.sol_d = np.hstack(ray.get([prob.get_solution_d.remote(
            ) for prob in self._subprob_cache.probs]))

            print('iter%d: end2end time %.4f, aug_lgr=%.4f' % (
                i, time.time() - start, aug_lgr))
        return sum(ray.get([
            prob.get_obj.remote() for prob in self._subprob_cache.probs]))

    def get_constr_dict(self, constrs):
        '''Get a mapping of constraint to its var_id_pos_list.'''
        constr_to_var_id_pos_list = {}
        for constr in constrs:
            constr_to_var_id_pos_list[
                constr] = get_var_id_pos_list_from_linear(constr.expr)
        return constr_to_var_id_pos_list

    def group_constrs(self, constrs, constr_dict):
        '''Group constraints into non-overlapped groups with union-find.'''
        parents = np.arange(len(constrs)).tolist()

        def find(x):
            if x == parents[x]:
                return x
            parents[x] = find(parents[x])
            return parents[x]

        def union(x1, x2):
            parent_x1 = find(x1)
            parent_x2 = find(x2)
            if parent_x1 != parent_x2:
                parents[parent_x2] = parent_x1

        var_id_pos_to_i = {}
        for i, constr in enumerate(constrs):
            for var_id_pos in constr_dict[constr]:
                if var_id_pos in var_id_pos_to_i:
                    union(var_id_pos_to_i[var_id_pos], i)
                var_id_pos_to_i[var_id_pos] = i

        parent_to_constrs = defaultdict(list)
        for i, parent in enumerate(parents):
            parent_to_constrs[find(parent)].append(constrs[i])
        return [constrs for _, constrs in parent_to_constrs.items()]

    def get_subproblems(self, num_cpus, rho):
        '''Return objective and constraints assignments for subproblems.'''

        # shuffle group order
        constrs_gps_idx_r = np.arange(len(self.constrs_gps_r))
        constrs_gps_idx_d = np.arange(len(self.constrs_gps_d))
        np.random.shuffle(constrs_gps_idx_r)
        np.random.shuffle(constrs_gps_idx_d)

        # get the set of var_id_pos
        var_id_pos_set_r = set()
        for constr, var_id_pos in self.constr_dict_r.items():
            var_id_pos_set_r.update(var_id_pos)
        var_id_pos_set_d = set()
        for constr, var_id_pos in self.constr_dict_d.items():
            var_id_pos_set_d.update(var_id_pos)

        # build actors with subproblems
        probs = []
        for cpu in range(num_cpus):
            # get constraint idx for the group
            idx_r = constrs_gps_idx_r[cpu::num_cpus].tolist()
            idx_d = constrs_gps_idx_d[cpu::num_cpus].tolist()
            # get constraints group
            constrs_r = [self.constrs_gps_r[j] for j in idx_r]
            constrs_d = [self.constrs_gps_d[j] for j in idx_d]
            # get obj groups
            obj_r = [self._obj_expr_r[j] for j in idx_r]
            obj_d = [self._obj_expr_d[j] for j in idx_d]
            # get var_id_to_pos_list
            var_id_to_pos_r = [[self.constr_dict_r[
                constr] for constr in constrs] for constrs in constrs_r]
            var_id_to_pos_d = [[self.constr_dict_d[
                constr] for constr in constrs] for constrs in constrs_d]
            # build subproblems
            probs.append(SubproblemsWrap.remote(
                idx_r, idx_d,
                obj_r, obj_d,
                constrs_r, constrs_d,
                var_id_to_pos_r, var_id_to_pos_d,
                var_id_pos_set_r, var_id_pos_set_d,
                rho))
        return probs

    def get_param_idx(self):
        '''Get parameter z index in last solution.'''
        # map var_id_pos in the big resource solution list
        sol_idx_r = ray.get([prob.get_solution_idx_r.remote(
        ) for prob in self._subprob_cache.probs])
        sol_idx_dict_r, idx = {}, 0
        for sol_idx in sol_idx_r:
            for var_id_pos in sol_idx:
                sol_idx_dict_r[var_id_pos] = idx
                idx += 1

        # map var_id_pos in the big demand solution list
        sol_idx_d = ray.get([prob.get_solution_idx_d.remote(
        ) for prob in self._subprob_cache.probs])
        sol_idx_dict_d, idx = {}, 0
        for sol_idx in sol_idx_d:
            for var_id_pos in sol_idx:
                sol_idx_dict_d[var_id_pos] = idx
                idx += 1

        # get parameter index
        param_idx_r = [[
            sol_idx_dict_d[var_id_pos] for var_id_pos in sol_idx
        ] for sol_idx in sol_idx_r]
        param_idx_d = [[
            sol_idx_dict_r[var_id_pos] for var_id_pos in sol_idx
        ] for sol_idx in sol_idx_d]
        return param_idx_r, param_idx_d

    def group_objective(self):
        '''Split objective into corresponding constraint groups'''
        var_id_pos_to_idx = defaultdict(list)
        for i, constrs_gps, constr_dict in zip(
            [0, 1],
            [self.constrs_gps_r, self.constrs_gps_d],
            [self.constr_dict_r, self.constr_dict_d]
        ):
            for j, constrs in enumerate(constrs_gps):
                for constr in constrs:
                    for var_id_pos in constr_dict[constr]:
                        var_id_pos_to_idx[var_id_pos].append((i, j))

        obj_r = [cp.Constant(0) for _ in self.constrs_gps_r]
        obj_d = [cp.Constant(0) for _ in self.constrs_gps_d]
        for obj in expand_expr(self.objective.expr):
            var_id_pos_list = get_var_id_pos_list_from_cone(obj)
            id_set = set(var_id_pos_to_idx[var_id_pos_list[0]])
            for var_id_pos in var_id_pos_list[1:]:
                id_set = id_set & set(var_id_pos_to_idx[var_id_pos])
            if not id_set:
                raise ValueError('Objective not separable.')
            idx = list(id_set)[0]
            if idx[0] == 0:
                obj_r[idx[1]] += obj
            else:
                obj_d[idx[1]] += obj

        return obj_r, obj_d
