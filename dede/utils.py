import cvxpy as cp
import numpy as np
from heapq import heappush, heappop

from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.atoms.affine.unary_operators import NegExpression
from cvxpy.atoms.affine.sum import Sum
from cvxpy.atoms.quad_over_lin import quad_over_lin
from cvxpy.expressions.variable import Variable
from cvxpy.expressions.leaf import Leaf
from cvxpy.expressions.constants import Constant


def expand_expr(expr):
    '''return a list of expanded expression
    TODO: add norm1, quad_form, convolve, multiply, MulExpression
    Args:
        expr: expression to expand
    '''
    if isinstance(expr, NegExpression):
        return [NegExpression(
            new_expr) for new_expr in expand_expr(expr.args[0])]
    elif isinstance(expr, AddExpression):
        expr_list = []
        for arg in expr.args:
            expr_list += expand_expr(arg)
        return expr_list
    elif isinstance(expr, Sum):
        return [Sum(arg) for arg in expr.args[0]]
    # (sum_{ij}X^2_{ij})/y
    elif isinstance(expr, quad_over_lin):
        return [quad_over_lin(arg, expr.args[1]) for arg in expr.args[0]]
    else:
        return [expr]


def replace_variables(expr, var_id_to_var):
    '''Replace variables in var_id_to_var with variables;
    Replace other variables with zero.
    Args:
        var_id_to_var: dictionary of var id to var
    '''
    if isinstance(expr, Constant):
        return expr
    elif isinstance(expr, AddExpression):
        args = expr._arg_groups
    else:
        args = expr.args
    data = expr.get_data()

    new_args = [arg for arg in args]
    for i, arg in enumerate(new_args):
        if isinstance(arg, Variable):
            new_args[i] = var_id_to_var[arg.id]
        elif not isinstance(arg, Leaf):
            new_args[i] = replace_variables(arg, var_id_to_var)

    if isinstance(expr, AddExpression):
        return type(expr)(new_args)
    elif data is not None:
        return type(expr)(*(new_args + data))
    else:
        return type(expr)(*new_args)


def get_var_id_pos_list_from_cone(expr):
    '''Return a list of (var_id, pos).'''
    # expr = cp.Variable(10, nonneg=True)

    data, _, _ = cp.Problem(
        cp.Minimize(expr)).get_problem_data(solver=cp.ECOS)

    col_to_var_id = {
        col: var_id for var_id, col in data[
            'param_prob'].var_id_to_col.items()}
    start_cols = sorted(col_to_var_id.keys()) + [len(data['c'])]
    active_var_id_set = {var.id for var in expr.variables()}
    num_zeros_nonneg = data['dims'].zero + data['dims'].nonneg

    var_id_pos_list = []
    start_col_i = 0

    for col, val in enumerate(data['c']):
        if not val:
            continue
        while col >= start_cols[start_col_i + 1]:
            start_col_i += 1
        start_col = start_cols[start_col_i]
        var_id = col_to_var_id[start_col]
        if var_id not in active_var_id_set:
            continue
        var_id_pos_list.append((var_id, col - start_col))

    G = data['G'].tocoo()
    for col in sorted(G.col[G.row >= num_zeros_nonneg]):
        while col >= start_cols[start_col_i + 1]:
            start_col_i += 1
        start_col = start_cols[start_col_i]
        var_id = col_to_var_id[start_col]
        if var_id not in active_var_id_set:
            continue
        var_id_pos_list.append((var_id, col - start_col))

    return var_id_pos_list


def get_var_id_pos_list_from_linear(expr):
    '''Return a list of (var_id, pos).'''
    data, _, _ = cp.Problem(
        cp.Minimize(expr.sum())).get_problem_data(solver=cp.ECOS)
    if data['dims'].zero or data['dims'].exp or data['dims'].soc \
            or data['dims'].psd or data['dims'].p3d:
        raise ValueError(f'Expression {expr} is not linear.')

    col_to_var_id = {
        col: var_id for var_id, col in data[
            'param_prob'].var_id_to_col.items()}
    start_cols = sorted(col_to_var_id.keys()) + [len(data['c'])]

    var_id_pos_list = []
    start_col_i = 0

    for col, val in enumerate(data['c']):
        if not val:
            continue
        while col >= start_cols[start_col_i + 1]:
            start_col_i += 1
        start_col = start_cols[start_col_i]
        var_id = col_to_var_id[start_col]
        var_id_pos_list.append((var_id, col - start_col))

    return var_id_pos_list


def heapsched_rt(lrts, k):
    '''Return a mathematical parallel runtime with k cpus for incoming jobs.'''
    h = []
    for rt in lrts[:k]:
        heappush(h, rt)

    curr_rt = 0
    for rt in lrts[k:]:
        curr_rt = heappop(h)
        heappush(h, rt + curr_rt)

    while len(h) > 0:
        curr_rt = heappop(h)

    return curr_rt


def parallelized_rt(lrts, k):
    '''Return a mathematical parallel runtime with k cpus for sorted jobs.'''
    if len(lrts) == 0:
        return 0.0
    inorder_rt = heapsched_rt(lrts, k)
    cp_bound = max(lrts)
    area_bound = sum(lrts) / k
    lrts.sort(reverse=True)
    two_approx = heapsched_rt(lrts, k)

    return two_approx
