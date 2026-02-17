import typing as t
from heapq import heappop, heappush

import cvxpy as cp
import numpy as np
from cvxpy import Parameter
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.atoms.affine.binary_operators import MulExpression, multiply
from cvxpy.atoms.affine.index import index
from cvxpy.atoms.affine.promote import Promote
from cvxpy.atoms.affine.sum import Sum
from cvxpy.atoms.affine.trace import trace
from cvxpy.atoms.affine.unary_operators import NegExpression
from cvxpy.atoms.elementwise.log import log
from cvxpy.atoms.quad_over_lin import quad_over_lin
from cvxpy.expressions.constants.constant import Constant
from cvxpy.expressions.leaf import Leaf
from cvxpy.expressions.variable import Variable
from numpy.typing import NDArray


class VarInfoT(t.NamedTuple):
    var_id: int  # id of the variable
    pos: int  # flattened position of the variable


BreakIntoVarsOutput = list[t.Union[bool, VarInfoT, "BreakIntoVarsOutput"]]


def expand_expr(expr: cp.Expression) -> list[cp.Expression]:
    """return a list of expanded expression
    TODO: add norm1, quad_form, convolve, multiply, MulExpression
    Args:
        expr: expression to expand
    """
    if isinstance(expr, (Variable, index, Constant, Parameter)):
        if len(expr.shape) == 0:
            return [expr]
        return [arg for arg in expr]
    elif isinstance(expr, NegExpression):
        return [NegExpression(new_expr) for new_expr in expand_expr(expr.args[0])]
    elif isinstance(expr, AddExpression):
        expr_list = []
        for arg in expr.args:
            expr_list += expand_expr(arg)
        return expr_list
    elif isinstance(expr, multiply):
        if len(expr.shape) == 0:
            return [expr]
        left_list = expand_expr(expr.args[0])
        right_list = expand_expr(expr.args[1])
        return [multiply(left, right) for left, right in zip(left_list, right_list)]
    elif isinstance(expr, MulExpression):
        if len(expr.shape) == 0:
            return [expr]
        return [expr[i] for i in range(expr.shape[0])]
    elif isinstance(expr, Sum):
        return [Sum(new_expr) for new_expr in expand_expr(expr.args[0])]
    # (sum_{ij}X^2_{ij})/y
    elif isinstance(expr, quad_over_lin):
        return [quad_over_lin(new_expr, expr.args[1]) for new_expr in expand_expr(expr.args[0])]
    elif isinstance(expr, log):
        return [log(new_expr) for new_expr in expand_expr(expr.args[0])]
    elif isinstance(expr, trace):
        return [expr.args[0][i, i] for i in range(expr.args[0].shape[0])]
    else:
        print(type(expr))
        return [expr]


def replace_variables(
    expr: cp.Expression, var_id_to_var: dict[int, NDArray[np.floating[t.Any]]]
) -> cp.Expression:
    """Replace variables in var_id_to_var with variables;
    Replace other variables with zero.
    Args:
        var_id_to_var: dictionary of var id to var
    """
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


def get_var_id_pos_list_from_linear(expr: cp.Expression) -> list[VarInfoT]:
    """
    Extracts variable ids and their corresponding positions from a expression.

    Return a list of (var_id, pos)."""
    terms = break_into_vars(expr)
    vars = set[VarInfoT]()

    def dfs(x: t.Union[bool, VarInfoT, BreakIntoVarsOutput]) -> None:
        if isinstance(x, list):
            for item in x:
                dfs(item)
        elif isinstance(x, tuple):
            vars.add(x)

    dfs(terms)
    return sorted(vars)


def break_into_vars(
    expr: cp.Expression,
) -> BreakIntoVarsOutput:
    """Helper for get_var_id_pos_list."""

    # Base case: constant reached
    if isinstance(expr, (int, float)):
        return [expr != 0]
    elif isinstance(expr, np.ndarray):
        return [bool(expr[idx] != 0) for idx in np.ndindex(expr.shape)]
    elif isinstance(expr, (Constant, Parameter)):
        assert expr.value is not None
        return [bool(expr.value[idx] != 0) for idx in np.ndindex(expr.value.shape)]

    # Base case: variable reached
    elif isinstance(expr, Variable):
        return [
            VarInfoT(expr.id, t.cast(int, np.ravel_multi_index(idx[::-1], expr.shape[::-1])))
            for idx in np.ndindex(expr.shape)
        ]

    # Base case: index object with underlying variable
    elif isinstance(expr, index) and isinstance(expr.args[0], Variable):
        var = expr.args[0]
        return [
            VarInfoT(var.id, t.cast(int, np.ravel_multi_index(idx[::-1], var.shape[::-1])))
            for idx in get_indices_from_index(expr)
        ]

    # Recursive case: index object with underlying expression
    elif isinstance(expr, index):
        all_vars = break_into_vars(expr.args[0])
        index_vars = []
        for idx in get_indices_from_index(expr):
            index_vars.append(all_vars[np.ravel_multi_index(idx, expr.args[0].shape)])
        return index_vars

    # Recursive case: promoted object
    elif isinstance(expr, Promote):
        return np.prod(expr.shape) * break_into_vars(expr.args[0])

    # Recursive case
    elif isinstance(expr, NegExpression):
        return break_into_vars(expr.args[0])
    elif isinstance(expr, Sum):
        return [break_into_vars(expr.args[0])]
    elif isinstance(expr, AddExpression):
        vars = [break_into_vars(arg) for arg in expr.args]
        return list(map(list, zip(*vars)))
    elif isinstance(expr, multiply):
        vars: list[t.Union[bool, VarInfoT, BreakIntoVarsOutput]] = []
        left_list = break_into_vars(expr.args[0])
        right_list = break_into_vars(expr.args[1])
        for left, right in zip(left_list, right_list):
            if left is False or right is False:
                vars.append(False)
                continue

            if isinstance(left, bool) and isinstance(right, bool):
                vars.append(True)
            elif isinstance(left, bool):
                vars.append(right)
            elif isinstance(right, bool):
                vars.append(left)
            else:
                vars.append(left + right)

        return vars
    else:
        raise TypeError("Unsupported expression type: {}".format(type(expr)))


def get_indices_from_index(index_obj: index) -> list[tuple[int, ...]]:
    """Returns all indices used in an index object."""
    key: list[slice] = index_obj.get_data()[0]  # key = (start, stop, step)

    shape: list[int] = []
    for k in key:
        # number of 'rows' is ceil((k.stop - k.start) / k.step)
        dim_size = (k.stop - k.start + k.step - 1) // k.step
        shape.append(dim_size)
    shape_tuple = tuple(shape)

    indices: list[tuple[int, ...]] = []
    for rel_idx in np.ndindex(shape_tuple):
        abs_idx: list[int] = []
        for axis, k in enumerate(key):
            abs_idx.append(k.start + rel_idx[axis] * k.step)
        indices.append(tuple(abs_idx))

    return indices


def get_var_id_pos_list_from_cone(expr: cp.Expression, solver: str) -> list[VarInfoT]:
    """Return a list of (var_id, pos)."""
    if not expr.variables():
        return []

    data, _, _ = cp.Problem(cp.Minimize(expr)).get_problem_data(solver=solver)

    col_to_var_id: dict[int, int] = {
        col: var_id for var_id, col in data["param_prob"].var_id_to_col.items()
    }
    start_cols = sorted(col_to_var_id.keys()) + [len(data["c"])]
    active_var_id_set: set[int] = {var.id for var in expr.variables()}
    num_zeros_nonneg: int = data["dims"].zero + data["dims"].nonneg

    var_id_pos_list: list[VarInfoT] = []
    start_col_i = 0

    for col, val in enumerate(t.cast(list[float], data["c"])):
        if not val:
            continue
        while col >= start_cols[start_col_i + 1]:
            start_col_i += 1
        start_col = start_cols[start_col_i]
        var_id = col_to_var_id[start_col]
        if var_id not in active_var_id_set:
            continue
        var_id_pos_list.append(VarInfoT(var_id, col - start_col))

    if data.get("G", None) is None:
        return var_id_pos_list

    G = data["G"].tocoo()
    for col in sorted(t.cast(list[int], G.col[G.row >= num_zeros_nonneg])):
        while col >= start_cols[start_col_i + 1]:
            start_col_i += 1
        start_col = start_cols[start_col_i]
        var_id = col_to_var_id[start_col]
        if var_id not in active_var_id_set:
            continue
        var_id_pos_list.append(VarInfoT(var_id, col - start_col))

    return var_id_pos_list


def heapsched_rt(lrts: list[float], k: int) -> float:
    """Return a mathematical parallel runtime with k cpus for incoming jobs."""
    h: list[float] = []
    for rt in lrts[:k]:
        heappush(h, rt)

    curr_rt = 0
    for rt in lrts[k:]:
        curr_rt = heappop(h)
        heappush(h, rt + curr_rt)

    while len(h) > 0:
        curr_rt = heappop(h)

    return curr_rt


def parallelized_rt(lrts: list[float], k: int) -> float:
    """Return a mathematical parallel runtime with k cpus for sorted jobs."""
    if len(lrts) == 0:
        return 0.0
    lrts.sort(reverse=True)
    two_approx = heapsched_rt(lrts, k)

    return two_approx
