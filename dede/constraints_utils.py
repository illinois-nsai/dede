import typing as t
from collections.abc import Iterable

import cvxpy as cp
from cvxpy import Parameter
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.atoms.affine.binary_operators import MulExpression, multiply
from cvxpy.atoms.affine.index import index
from cvxpy.atoms.affine.promote import Promote
from cvxpy.atoms.affine.sum import Sum
from cvxpy.atoms.affine.unary_operators import NegExpression
from cvxpy.expressions.constants.constant import Constant
from cvxpy.expressions.variable import Variable

DirT = t.Union[t.Literal[0], t.Literal[1]]


def breakdown_constr(
    constr: t.Union[cp.Constraint, t.Iterable[cp.Constraint]], dir: DirT
) -> list[cp.Constraint]:
    if isinstance(constr, Iterable):
        out = []
        for c in constr:
            out.extend(breakdown_constr(c, dir))
        return out

    return [expr == 0 for expr in breakdown_expression(constr.expr, dir)]


def split_index_by_dir(index_obj: index, dir: DirT) -> list[cp.Expression]:
    """Given a multi-dimensional (2d) index object and a direction, splits the index
    into a list of one-dimensional indices, one for each index along the specified direction,
    and returns the resulting Expressions."""
    key: tuple[slice, ...] = index_obj.get_data()[0]
    var: cp.Expression = index_obj.args[0]

    split_list = []

    new_key: list[t.Union[int, slice]] = list(key)
    k = t.cast(slice, key[dir])
    for i in range(k.start, k.stop, k.step):
        new_key[dir] = i
        split_list.append(var[tuple(new_key)])

    return split_list


def breakdown_expression(expr: cp.Expression, dir: DirT) -> list[cp.Expression]:
    if len(expr.shape) > 2:
        raise TypeError("3D constraints and above are not supported")

    # Base cases
    if isinstance(expr, Constant):
        if len(expr.value.shape) <= 1:
            return [Constant(expr.value)]
        elif dir == 0:
            return [Constant(expr.value[i, :]) for i in range(expr.value.shape[0])]
        else:
            return [Constant(expr.value[:, j]) for j in range(expr.value.shape[1])]

    elif isinstance(expr, (Variable, Parameter)):
        if len(expr.shape) <= 1:
            return [expr]
        elif dir == 0:
            return [expr[i, :] for i in range(expr.shape[0])]
        else:
            return [expr[:, j] for j in range(expr.shape[1])]

    elif isinstance(expr, index):
        if len(expr.args[0].shape) <= 1:
            return [expr]
        else:
            return split_index_by_dir(expr, dir)

    elif isinstance(expr, Promote):
        if len(expr.shape) <= 1:
            return [expr]
        else:
            expr_list_len = expr.shape[0] if dir == 0 else expr.shape[1]
            return expr_list_len * [expr.args[0]]

    # Recursive cases
    elif isinstance(expr, NegExpression):
        return [NegExpression(subexpr) for subexpr in breakdown_expression(expr.args[0], dir)]

    elif isinstance(expr, Sum):
        return [expr]

    elif isinstance(expr, AddExpression):
        terms = []
        expr_list = []
        for arg in expr.args:
            expr_list.append(breakdown_expression(arg, dir))
        for j in range(len(expr_list[0])):
            term = 0
            for i in range(len(expr_list)):
                term += expr_list[i][j]
            terms.append(term)
        return terms

    elif isinstance(expr, multiply):
        left_list = breakdown_expression(expr.args[0], dir)
        right_list = breakdown_expression(expr.args[1], dir)
        return [multiply(left_list[i], right_list[i]) for i in range(len(left_list))]

    elif isinstance(expr, MulExpression):
        return [expr]

    else:
        raise TypeError(f"Expression Not Supported: {type(expr)}")
