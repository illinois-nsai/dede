#!/usr/bin/env python3

import numpy as np
import cvxpy as cp
from dede.constraints_utils import func

def test_single_var():
    x = cp.Variable()
    constr = 2 * x == 1
    out = func(constr)
    expected = [constr]

    assert tester(out) == tester(expected)

def test_1d_matrix():
    x = cp.Variable(4)

    # test element by element
    constr = x @ np.arange(1, 5) >= 1
    out = func(constr)
    expected = [x[i] * (i + 1) >= 1 for i in range(4)]

    assert tester(out) == tester(expected)

    # test scalar multiplication
    constr = x * 2 == 1
    out = func(constr)
    expected = [x[i] * 2 == 1 for i in range(4)]

    assert tester(out) == tester(expected)

def test_2d_matrix():
    x = cp.Variable((4, 3))

    # test matrix multiplication 2d x 2d
    y = np.arange(1, 7).reshape((3, 2))
    constr = x @ y >= 1
    out = func(constr)
    expected = [
        x[0, 0] * 1 + x[0, 1] * 3 + x[0, 2] * 5 >= 1,
        x[0, 0] * 2 + x[0, 1] * 4 + x[0, 2] * 6 >= 1,
        x[1, 0] * 1 + x[1, 1] * 3 + x[1, 2] * 5 >= 1,
        x[1, 0] * 2 + x[1, 1] * 4 + x[1, 2] * 6 >= 1,
        x[2, 0] * 1 + x[2, 1] * 3 + x[2, 2] * 5 >= 1,
        x[2, 0] * 2 + x[2, 1] * 4 + x[2, 2] * 6 >= 1,
        x[3, 0] * 1 + x[3, 1] * 3 + x[3, 2] * 5 >= 1,
        x[3, 0] * 2 + x[3, 1] * 4 + x[3, 2] * 6 >= 1
    ]

    assert tester(out) == tester(expected)

    # test matrix multiplication 2d x 1d
    y = np.arange(1, 4)
    constr = x @ y >= 1
    out = func(constr)
    expected = [x[i, 0] * 1 + x[i, 1] * 2 + x[i, 2] * 3 >= 1 for i in range(4)]

    assert tester(out) == tester(expected)

    # test scalar multiplication
    constr = x * 2 == 1
    out = func(constr)
    expected = [x[tup] * 2 == 1 for tup in np.ndindex(x.shape)]

    assert tester(out) == tester(expected)

def test_hard2(): # fix this
    x = cp.Variable((4, 4))
    constr = [x[0, 2:].sum() + x[2, 3] >= 10, x[2:] @ np.arange(5,9) >= 0]
    out = func(constr)

    for t in out:
        print(t)

def tester(constr):
    return [str(c) for c in constr]

if __name__ == '__main__':
    test_single_var()
    test_1d_matrix()
    test_2d_matrix()
    test_hard2()