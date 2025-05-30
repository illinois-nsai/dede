import cvxpy as cp
import numpy as np
from cvxpy.constraints.zero import Equality
from cvxpy.constraints.nonpos import Inequality
from cvxpy.atoms.affine.sum import Sum
from cvxpy.expressions.constants.constant import Constant
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.atoms.affine.binary_operators import MulExpression
from cvxpy.expressions.variable import Variable

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
    print("Left side:", left)
    print("Right side:", right)
    print("Type of left side:", type(left))
    print("Type of right side:", type(right))
    print("left value:", left.value)
    print("Type of left value:", type(left.value))
    print("right value:", right.value)
    print("Type of right value:", type(right.value))
    if type(constr) == Inequality or type(constr) == Equality:
        if type(left) == Sum:
            for i in range(left.shape[0]):
                left_list.append(left.args[0][i])
        elif type(left) == Variable:
            left_list.append(left)
        elif type(left.value) == np.ndarray:
            print("Left shape", left.shape)
            for idx in np.ndindex(left.shape):
                left_list.append(left[idx])
        elif left.value != None and type(float(left.value)) == float:
            left_list.append(float(left.value))
        elif type(left.value) == list or type(left.value) == np.ndarray:
            for v in left.value:
                left_list.append(float(v))
        # elif type(left) == AddExpression:
        #     need to finish
        else:
            raise TypeError(f"Left side of the constraint is not a valid type: {type(left)}")
        
        if type(right) == Sum:
            for i in range(right.shape[0]):
                right_list.append(right.args[0][i])
        elif type(right) == Variable:
            right_list.append(right)
        elif type(right.value) == np.ndarray:
            for idx in np.ndindex(right.shape):
                right_list.append(right[idx])  
        elif right.value != None and type(float(right.value)) == float:
            right_list.append(float(right.value))
        elif type(list(right.value)) == list or type(right.value) == np.ndarray:
            for v in right.value:
                right_list.append(float(v))
        # elif type(right) == AddExpression:
        #     Need to finish
        else:
            raise TypeError(f"Right side of the constraint is not a valid type: {type(right)}")
    else:
        raise TypeError("Only Inequality or Equality based Testing.")
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
