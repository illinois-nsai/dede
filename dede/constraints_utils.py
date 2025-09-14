import numpy as np
import cvxpy as cp
from cvxpy.constraints.zero import Equality
from cvxpy.constraints.nonpos import Inequality
from cvxpy.atoms.affine.sum import Sum
from cvxpy.expressions.constants.constant import Constant
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.expressions.variable import Variable
from cvxpy.atoms.affine.index import index
from cvxpy.atoms.affine.promote import Promote
from cvxpy.atoms.affine.binary_operators import MulExpression
from cvxpy.atoms.affine.binary_operators import multiply
from cvxpy.atoms.affine.unary_operators import NegExpression


def breakdown_constr(constr):
    if isinstance(constr, (list, tuple)):
        out = []
        for c in constr:
            out.extend(breakdown_constr(c))
        return out

    expr_list = breakdown_expression(constr.expr)

    constr_list = []
    for expr in expr_list:
        constr_list.append(expr == 0)

    return constr_list


def get_indices_from_index(index_obj):
    '''Returns all indices used in an index object.'''
    key = index_obj.get_data()[0]  # key = (start, stop, step)

    shape = []
    for k in key:
        # number of 'rows' is ceil((k.stop - k.start) / k.step)
        dim_size = (k.stop - k.start + k.step - 1) // k.step
        shape.append(dim_size)
    shape = tuple(shape)

    indices = []
    for rel_idx in np.ndindex(shape):
        abs_idx = []
        for axis, k in enumerate(key):
            abs_idx.append(k.start + rel_idx[axis] * k.step)
        indices.append(tuple(abs_idx))

    return indices


def split_index_by_dir(index_obj, dir):
    key = index_obj.get_data()[0]
    var = index_obj.args[0]

    split_list = []

    new_key = list(key)
    k = key[dir]
    for i in range(k.start, k.stop, k.step):
        new_key[dir] = i
        split_list.append(var[tuple(new_key)])
    
    return split_list


def breakdown_expression(expr, dir):
    # Base cases
    if isinstance(expr, Constant):
        if len(expr.value.shape) <= 1:
            return [expr.value]
        elif dir == 0:
            return [expr.value[i, :] for i in range(expr.value.shape[0])]
        else:
            return [expr.value[:, j] for j in range(expr.value.shape[1])]
    
    elif isinstance(expr, Variable):
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
    
    # Recursive cases
    elif isinstance(expr, Promote):
        if len(expr.shape) <= 1:
            return [expr]
        else:
            expr_list_len = expr.shape[0] if dir == 0 else expr.shape[1]
            return expr_list_len * [expr.args[0]]

    elif isinstance(expr, NegExpression):
        return [NegExpression(subexpr) for subexpr in breakdown_expression(expr.args[0], dir)]
    elif isinstance(expr, Sum):
        # TODO: add axis
        '''
        term = 0
        for subexpr in breakdown_expression(expr.args[0]):
            term += subexpr
        return [term]
        '''
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
        return [left_list[i] * right_list[i] for i in range(len(left_list))]
    
    else:
        raise TypeError(f"Expression Not Supported: {type(expr)}")


'''
def breakdown_expression(expr):
    terms = []

    if isinstance(expr, Sum):
        if expr.axis == None: # sum over all elements
            terms.append(expr)
        elif expr.axis == 0: # 0 for summing over columns 
            for i in range(expr.shape[0]):
                terms.append(expr.args[0][:, i])
        elif expr.axis == 1: # 1 for summing over rows
            for i in range(expr.shape[0]):
                terms.append(expr.args[0][i])

    elif isinstance(expr, AddExpression):
        expr_list = []
        for var in expr.args:
            expr_list.append(breakdown_expression(var))
        for j in range(len(expr_list[0])):
            var = 0
            for i in range(len(expr_list)):
                var = var + expr_list[i][j]
            terms.append(var)
    
    elif isinstance(expr, MulExpression):
        left, right = expr.args

        left_list = breakdown_expression(left)
        right_list = breakdown_expression(right)

        # scalar multiplication
        if isinstance(left, Promote) or isinstance(right, Promote):
            for i in range(len(left_list)):
                terms.append(left_list[i] * right_list[i])
        # element by element
        elif len(left.shape) <= 1 and left.shape == right.shape:
            for i in range(len(left_list)):
                terms.append(left_list[i] * right_list[i])
        # matrix multiplication
        else:
            left_shape = left.shape # N x K
            right_shape = right.shape # K x M

            if len(left_shape) == 1:
                left_shape = (1, left_shape[0])
            elif len(right_shape) == 1:
                if right_shape[0] == left_shape[1]:
                    right_shape = (right_shape[0], 1)
                else:
                    right_shape = (1, right_shape[0])

            N, M, K = left_shape[0], right_shape[1], left_shape[1]
            for i in range(N):
                for j in range(M):
                    sum = 0
                    for k in range(K):
                        sum += left_list[i * K + k] * right_list[k * M + j]
                    terms.append(sum) 

    elif isinstance(expr, Variable):
        if expr.shape == ():
            terms.append(expr)
        else:
            # get index object first
            index_obj = expr[tuple(slice(None) for _ in expr.shape)]
            terms = get_entries_from_index(index_obj)

    elif isinstance(expr, Constant):
        for i in np.ndindex(expr.value.shape):
            terms.append(float(expr.value[i]))

    elif isinstance(expr, index):
        terms = get_entries_from_index(expr)

    elif isinstance(expr, Promote):
        inner_term = breakdown_expression(expr.args[0])[0]
        for rel_idx in np.ndindex(expr.shape):
            terms.append(inner_term)

    else:
        raise TypeError(f"Expression Not Supported: {type(expr)}")
    
    return terms
'''
