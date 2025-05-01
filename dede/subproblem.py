import numpy as np
import cvxpy as cp
import time
from collections import defaultdict
from scipy.sparse import coo_matrix

from cvxpy.problems.problem import Problem as CpProblem
import cvxpy.lin_ops.lin_utils as lu

from .utils import replace_variables


class Subproblem(CpProblem):
    '''Subproblem for per-resource allocation or per-demand allocation.'''

    def __init__(
            self, idx, objective_expr, constrs_gp,
            active_var_id_to_pos_gp, inactive_var_id_pos_set, rho):
        '''Initialize subproblem.
        Args:
            idx: subproblem id. first ele represent resource or demand problem.
            objective_expr: expression for objective function
            constrs_gp: a list of constraints
            active_var_id_to_pos_gp: (var_id, position) list of constraints
            inactive_var_id_pos_set: variables in the z round of problems
            rho: rho value in ADMM; 1 if None
        '''

        self.id = idx
        self.rho = rho

        # create var for the subproblem
        var_id_pos_set1, var_id_pos_set2 = set(), set()
        for var_id_pos_list in active_var_id_to_pos_gp:
            for var_id_pos in var_id_pos_list:
                # whether var_id_pos has its counterpart in inactive var_id_pos
                if var_id_pos in inactive_var_id_pos_set:
                    var_id_pos_set1.add(var_id_pos)
                else:
                    var_id_pos_set2.add(var_id_pos)
        self.var_id_pos_list = sorted(list(var_id_pos_set1)) + \
            sorted(list(var_id_pos_set2))
        self.x_z_num = len(var_id_pos_set1)

        # avoid parameters of the same id in the same process
        max_param_id = -1
        for param in objective_expr.parameters():
            max_param_id = max(max_param_id, param.id)
        for constr in constrs_gp:
            for param in constr.parameters():
                max_param_id = max(max_param_id, param.id)
        param_id = lu.get_id()
        while param_id < max_param_id:
            param_id = lu.get_id()

        # initiate var and param
        self.var = cp.Variable(len(self.var_id_pos_list))
        self.var.value = np.zeros(self.var.shape)
        self.param = cp.Parameter(self.x_z_num)
        self.param.value = np.zeros(self.param.shape)

        # replace inactive var with 0
        var_id_to_new_var, constrs_var_attr = self.get_var_id_to_new_var(
            objective_expr, constrs_gp)

        # objective
        self.obj_expr_old = replace_variables(
            objective_expr, var_id_to_new_var)

        # lambda for original constraints
        self.f1 = cp.hstack([replace_variables(
            constr.expr, var_id_to_new_var
        ).flatten() for constr in constrs_gp])
        self.l1 = cp.Parameter(self.f1.shape, value=np.zeros(self.f1.shape))
        # lambda for x = z
        self.f2 = self.var[:self.x_z_num] - self.param
        self.l2 = cp.Parameter(self.f2.shape, value=np.zeros(self.f2.shape))

        super(Subproblem, self).__init__(
            cp.Minimize(
                self.obj_expr_old +
                self.rho / 2 * cp.sum_squares(self.f1 + self.l1) +
                self.rho / 2 * cp.sum_squares(self.f2 + self.l2)),
            constrs_var_attr)

    def get_var_id_to_new_var(self, objective_expr, constrs_gp):
        '''Replace inactive var in old var.'''
        var_id_to_var = {}
        for constr in [objective_expr] + constrs_gp:
            for var in constr.variables():
                var_id_to_var[var.id] = var

        var_id_to_pos = defaultdict(list)
        for var_id, pos in self.var_id_pos_list:
            var_id_to_pos[var_id].append(pos)

        var_id_pos_to_idx = {k: i for i, k in enumerate(self.var_id_pos_list)}

        var_id_to_new_var = {}
        constrs_var_attr = []
        attr_idx = {'nonneg': [], 'boolean': [], 'integer': []}
        for var_id, var in var_id_to_var.items():

            # fill pos not in pos_list with 0
            pos_list = var_id_to_pos[var_id]
            m = coo_matrix((
                np.ones(len(pos_list)),
                (
                    pos_list,
                    [var_id_pos_to_idx[(var_id, pos)] for pos in pos_list])
            ), shape=(var.size, self.var.size))
            var_id_to_new_var[var.id] = (m @ self.var).reshape(var.shape)

            # collect variable attributes
            for attr, value in var.attributes.items():
                if not value:
                    continue
                for pos in pos_list:
                    idx = var_id_pos_to_idx[(var_id, pos)]
                    if attr in ['nonneg', 'boolean', 'integer']:
                        attr_idx[attr].append(idx)
                    else:
                        raise ValueError(f'{attr} is not supported.')
        if attr_idx['nonneg']:
            constrs_var_attr.append(self.var[attr_idx['nonneg']] >= 0)
        if attr_idx['boolean']:
            constrs_var_attr.append(
                self.var[attr_idx['boolean']] == cp.Variable(
                    len(attr_idx['boolean']), boolean=True))
        if attr_idx['integer']:
            constrs_var_attr.append(
                self.var[attr_idx['integer']] == cp.Variable(
                    len(attr_idx['integer']), integer=True))

        return var_id_to_new_var, constrs_var_attr

    def get_solution_idx(self):
        '''Record (var_id, position) to keep similar in the z round.'''
        return self.var_id_pos_list[:self.x_z_num]

    def get_solution(self):
        '''Return solution.'''
        return self.var.value[:self.x_z_num]

    def get_obj(self):
        '''Return value of the original objective function.'''
        return self.obj_expr_old.value

    def solve(self, param_value, *args, **kwargs):
        '''Update lambda and solve the subproblem.
        Args:
        param_value: value of parameters that keeps x == z
        '''
        self.l1.value += self.f1.value
        if self.id[0] == 1:
            # note: we haven't update param yet
            self.l2.value += self.f2.value
        self.param.value = param_value
        if self.id[0] == 0:
            self.l2.value += self.f2.value

        return super(Subproblem, self).solve(*args, **kwargs)
