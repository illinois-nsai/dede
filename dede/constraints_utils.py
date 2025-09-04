#!/usr/bin/env python3

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
from cvxpy.atoms.affine.unary_operators import NegExpression


def breakdown_constr(constr, dir):
    if isinstance(constr, (list, tuple)):
        out = []
        for c in constr:
            out.extend(breakdown_constr(c, dir))
        return out

    expr_list = breakdown_expression(constr.expr, dir)

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


def breakdown_expression(expr, dir):
    # Base cases
    if isinstance(expr, Constant):
        return [expr.value[idx] for idx in np.ndindex(expr.value.shape)]
    
    elif isinstance(expr, Variable):
        index_obj = expr[tuple(slice(None) for _ in expr.shape)]
        indices = get_indices_from_index(index_obj)
        return [expr[idx] for idx in indices]

    elif isinstance(expr, index) and isinstance(expr.args[0], Variable):
        indices = get_indices_from_index(expr)
        var = expr.args[0]
        return [var[idx] for idx in indices]

    # Recursive cases
    elif isinstance(expr, index):
        all_terms = breakdown_expression(expr.args[0], dir)
        index_terms = []
        for idx in get_indices_from_index(expr):
            index_terms.append(all_terms[np.ravel_multi_index(idx, expr.args[0].shape)])
        return index_terms

    elif isinstance(expr, Promote):
        return np.prod(expr.shape) * breakdown_expression(expr.args[0], dir)

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
        axis = expr.axis
        inner = expr.args[0]

        if dir == "row":
            if axis is None:
                # For axis=None, decompose only when inner is an index over a Variable
                # that selects a partial multi-element slice along any axis.
                if isinstance(inner, index) and isinstance(inner.args[0], Variable):
                    key = inner.get_data()[0]
                    var_shape = inner.args[0].shape
                    def slice_len(k):
                        return (k.stop - k.start + k.step - 1) // k.step
                    for i, k in enumerate(key):
                        if isinstance(k, slice):
                            length_i = slice_len(k)
                            # Don't break down stride-based slices (step != 1)
                            if length_i > 1 and length_i != var_shape[i] and k.step == 1:
                                return breakdown_expression(inner, dir)
                return [expr]
            elif axis == 1:
                return [cp.sum(inner[i, :]) for i in range(inner.shape[0])]
            elif axis == 0:
                summed = cp.sum(inner, axis=0)
                return [summed[i] for i in range(summed.shape[0])]

        elif dir == "col":
            if axis is None:
                if isinstance(inner, index) and isinstance(inner.args[0], Variable):
                    key = inner.get_data()[0]
                    var_shape = inner.args[0].shape
                    def slice_len(k):
                        return (k.stop - k.start + k.step - 1) // k.step
                    for i, k in enumerate(key):
                        if isinstance(k, slice):
                            length_i = slice_len(k)
                            # Don't break down stride-based slices (step != 1)
                            if length_i > 1 and length_i != var_shape[i] and k.step == 1:
                                return breakdown_expression(inner, dir)
                return [expr]
            elif axis == 0:
                return [cp.sum(inner[:, j]) for j in range(inner.shape[1])]
            elif axis == 1:
                summed = cp.sum(inner, axis=1)
                return [summed[j] for j in range(summed.shape[0])]

        else:
            term = 0
            for subexpr in breakdown_expression(inner, dir):
                term += subexpr
            return [term]
        return [expr]
    elif isinstance(expr, AddExpression):
        terms = []
        expr_list = []
        for arg in expr.args:
            expr_list.append(breakdown_expression(arg, dir))
        
        # Find the maximum length among all argument breakdowns
        max_len = max(len(exprs) for exprs in expr_list)
        
        for j in range(max_len):
            term = 0
            for i in range(len(expr_list)):
                # Use the j-th element if it exists, otherwise use 0
                if j < len(expr_list[i]):
                    term += expr_list[i][j]
                else:
                    # For constants, we can add 0, but for expressions we need to be more careful
                    if isinstance(expr_list[i][0], (int, float)):
                        term += 0
                    else:
                        # For non-constant expressions, we need to add a zero expression
                        # This is a bit tricky, but for now let's assume it's a constant
                        term += 0
            terms.append(term)
        return terms
    elif isinstance(expr, multiply):
        left_list = breakdown_expression(expr.args[0], dir)
        right_list = breakdown_expression(expr.args[1], dir)
        return [left[i] * right[i] for i in range(len(left_list))]
    
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
