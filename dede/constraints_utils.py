import numpy as np
from cvxpy.constraints.zero import Equality
from cvxpy.constraints.nonpos import Inequality
from cvxpy.atoms.affine.sum import Sum
from cvxpy.expressions.constants.constant import Constant
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.expressions.variable import Variable
from cvxpy.atoms.affine.index import index
from cvxpy.atoms.affine.promote import Promote


def func(constr):
    if isinstance(constr, (list, tuple)):
        out = []
        for c in constr:
            out.extend(func(c))
        return out
    left, right = constr.args
    constr_list = []
    left_list = []
    right_list = []
    if isinstance(constr, (Equality, Inequality)):
        if isinstance(left, Sum):
            if left.axis == 0:
                for i in range(left.shape[0]):
                    left_list.append(left.args[0][:, i])
            elif left.axis == 1:
                for i in range(left.shape[0]):
                    left_list.append(left.args[0][i])
        elif isinstance(left, AddExpression):
            expr_list = []
            for expr in left.args:
                expr_list.extend(breakdown_expression(expr))
            combined_left = expr_list[0]
            for t in expr_list[1:]:
                combined_left = combined_left + t
            left_list.append(combined_left)
        elif isinstance(left, Variable):
            left_list.append(left)
        elif isinstance(left, Constant) and isinstance(left.value, np.ndarray):
            for i in np.ndindex(left.value.shape):
                left_list.append(float(left.value[i]))
        elif isinstance(left.value, np.ndarray) or isinstance(left, index):
            for i in np.ndindex(left.shape):
                left_list.append(left[i])  # Cast to Float if Needed
        elif left.value is not None and isinstance(float(left.value), float):
            left_list.append(float(left.value))
        elif isinstance(list(left.value), (list, np.ndarray)):
            for v in left.value:
                left_list.append(float(v))
        else:
            raise TypeError(f"Left Expression Not Supported: {type(left)}")

        if isinstance(right, Sum):
            for i in range(right.shape[0]):
                right_list.append(right.args[0][i])
        elif isinstance(right, AddExpression):
            expr_list = []
            for expr in right.args:
                expr_list.extend(breakdown_expression(expr))
            combined_right = expr_list[0]
            for t in expr_list[1:]:
                combined_right = combined_right + t
            right_list.append(combined_right)
        elif isinstance(right, Variable):
            right_list.append(right)
        elif isinstance(right, Constant) and isinstance(right.value, np.ndarray):
            for i in np.ndindex(right.value.shape):
                right_list.append(float(right.value[i]))
        elif isinstance(right.value, np.ndarray) or isinstance(right, index):
            for i in np.ndindex(right.shape):
                right_list.append(right[i])  # Cast to Float if Needed
        elif right.value is not None and isinstance(float(right.value), float):
            right_list.append(float(right.value))
        elif isinstance(list(right.value), list):
            for v in right.value:
                right_list.append(float(v))
        else:
            raise TypeError(f"Right Expression Not Supported: {type(right)}")
    else:
        raise TypeError("Only Inequality or Equality based Testing.")

    if isinstance(constr, Inequality):
        if len(left_list) < len(right_list):
            for i in range(len(right_list)):
                constr_list.append(left_list[0] <= right_list[i])
        elif len(left_list) > len(right_list):
            for i in range(len(left_list)):
                constr_list.append(left_list[i] <= right_list[0])
        elif len(left_list) == len(right_list):
            for i in range(len(left_list)):
                constr_list.append(left_list[i] <= right_list[i])

    elif isinstance(constr, Equality):
        if len(left_list) < len(right_list):
            for i in range(len(right_list)):
                constr_list.append(left_list[0] == right_list[i])
        elif len(left_list) > len(right_list):
            for i in range(len(left_list)):
                constr_list.append(left_list[i] == right_list[0])
        elif len(left_list) == len(right_list):
            for i in range(len(left_list)):
                constr_list.append(left_list[i] == right_list[i])

    return constr_list


def breakdown_expression(expr):
    expr_list = []
    if isinstance(expr, Sum):
        for i in range(expr.shape[0]):
            expr_list.append(expr.args[0][i])
    elif isinstance(expr, AddExpression):
        for addend in expr.args:
            subterms = breakdown_expression(addend)
            expr_list.extend(subterms)
    elif isinstance(expr, Variable):
        expr_list.append(expr)
    elif isinstance(expr, Promote):
        expr_list.extend(breakdown_expression(expr.args[0]))
    elif isinstance(expr.value, np.ndarray) or isinstance(expr, index):
        for i in np.ndindex(expr.shape):
            expr_list.append(expr[i])  # Cast to Float if Needed
    elif expr.value is not None and isinstance(float(expr.value), float):
        expr_list.append(float(expr.value))
    elif isinstance(expr.value, (list, np.ndarray)):
        for v in expr.value:
            expr_list.append(float(v))
    else:
        raise TypeError(f"In Breakdown Expression: {type(expr)}")

    return expr_list
