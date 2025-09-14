#!/usr/bin/env python3

import dede as dd
import cvxpy as cp
import numpy as np
from scipy.optimize import minimize
import math
from dede.constraints_utils import breakdown_expression


def test():
    x = dd.Variable((5, 5))
    expr = x[0:2, 2:5:2]
    expr = x[0:4, 2:5] * 3
    dir = 1
    for expr in breakdown_expression(expr, dir):
        print(expr)


if __name__ == '__main__':
    test()