import cvxpy as cp
import numpy as np
from cvxpy.constraints.zero import Equality
from cvxpy.constraints.nonpos import Inequality
from cvxpy.atoms.affine.sum import Sum
from cvxpy.expressions.constants.constant import Constant
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.atoms.affine.binary_operators import MulExpression
from cvxpy.expressions.variable import Variable
from cvxpy.atoms.affine.index import index
from cvxpy.atoms.affine.promote import Promote

def func(constr):
    if type(constr) == list or type(constr) == tuple:
        out = []
        for c in constr:
            out.extend(func(c))
        return out
    left, right = constr.args
    constr_list = []
    left_list = []
    right_list = []
    # print("Left side:", left)
    # print("Right side:", right)
    # print("Type of left side:", type(left))
    # print("Type of right side:", type(right))
    # print("left value:", left.value)
    # print("Type of left value:", type(left.value))
    # print("right value:", right.value)
    # print("Type of right value:", type(right.value))
    if type(constr) == Inequality or type(constr) == Equality:
        if type(left) == Sum:
            if left.axis == 0:
                for i in range(left.shape[0]):
                    left_list.append(left.args[0][:, i])
            elif left.axis == 1:
                for i in range(left.shape[0]):
                    left_list.append(left.args[0][i])
        elif type(left) == AddExpression:
            expr_list = []
            for expr in left.args:
                expr_list.extend(breakdown_expression(expr))
            combined_left = expr_list[0]
            for t in expr_list[1:]:
                combined_left = combined_left + t
            left_list.append(combined_left)
        
        elif type(left) == Variable:
            left_list.append(left)
        elif type(left) == Constant and type(left.value) == np.ndarray:
            for i in np.ndindex(left.value.shape):
                left_list.append(float(left.value[i]))
        elif type(left.value) == np.ndarray or type(left) == index:
            for i in np.ndindex(left.shape):
                left_list.append(left[i]) # if it is a float, then after the fact maybe cast it to a float
        elif left.value != None and type(float(left.value)) == float:
            left_list.append(float(left.value))
        elif type(left.value) == list or type(left.value) == np.ndarray:
            for v in left.value:
                left_list.append(float(v))
        else:
            raise TypeError(f"Left side of the constraint is not a valid type: {type(left)}")
        
        if type(right) == Sum:
            for i in range(right.shape[0]):
                right_list.append(right.args[0][i])
        elif type(right) == AddExpression:
            expr_list = []
            for expr in right.args:
                expr_list.extend(breakdown_expression(expr))
            combined_right = expr_list[0]
            for t in expr_list[1:]:
                combined_right = combined_right + t
            right_list.append(combined_right)
        elif type(right) == Variable:
            right_list.append(right)
        elif type(right) == Constant and type(right.value) == np.ndarray:
            for i in np.ndindex(right.value.shape):
                right_list.append(float(right.value[i]))
        elif type(right.value) == np.ndarray or type(right) == index:
            for i in np.ndindex(right.shape):
                right_list.append(right[i])  # if it is a float, then after the fact maybe cast it to a float
        elif right.value != None and type(float(right.value)) == float:
            right_list.append(float(right.value))
        elif type(list(right.value)) == list or type(right.value) == np.ndarray:
            for v in right.value:
                right_list.append(float(v))
        else:
            raise TypeError(f"Right side of the constraint is not a valid type: {type(right)}")
    else:
        raise TypeError("Only Inequality or Equality based Testing.")
    print("Left List:", left_list)
    print("Right List:", right_list)
    if len(left_list) < len(right_list) and type(constr) == Inequality:
        for i in range(len(right_list)):
            constr_list.append(left_list[0] <= right_list[i])
    elif len(left_list) > len(right_list) and type(constr) == Inequality:
        for i in range(len(left_list)):
            constr_list.append(left_list[i] <= right_list[0])
    elif len(left_list) == len(right_list) and type(constr) == Inequality:
        for i in range(len(left_list)):
            constr_list.append(left_list[i] <= right_list[i])
    elif len(left_list) < len(right_list) and type(constr) == Equality:
        for i in range(len(right_list)):
            constr_list.append(left_list[0] == right_list[i])
    elif len(left_list) > len(right_list) and type(constr) == Equality:
        for i in range(len(left_list)):
            constr_list.append(left_list[i] == right_list[0])
    elif len(left_list) == len(right_list) and type(constr) == Equality:
        for i in range(len(left_list)):
            constr_list.append(left_list[i] == right_list[i])
    return constr_list

def breakdown_expression(expr):
    expr_list = []
    if type(expr) == Sum:
        for i in range(expr.shape[0]):
            expr_list.append(expr.args[0][i])
    elif type(expr) == AddExpression:
            for addend in expr.args:
                subterms = breakdown_expression(addend)
                expr_list.extend(subterms)
    elif type(expr) == Variable:
        expr_list.append(expr)
    elif type(expr) == Promote:
        expr_list.extend(breakdown_expression(expr.args[0]))
    elif type(expr.value) == np.ndarray or type(expr) == index:
        for i in np.ndindex(expr.shape):
            expr_list.append(expr[i]) # if it is a float, then after the fact maybe cast it to a float
    elif expr.value != None and type(float(expr.value)) == float:
        expr_list.append(float(expr.value))
    elif type(expr.value) == list or type(expr.value) == np.ndarray:
        for v in expr.value:
            expr_list.append(float(v))
    else:
        raise TypeError(f"In Breakdown Expression: {type(expr)}")
    return expr_list