#!/usr/bin/env python3

import numpy as np
import cvxpy as cp
from dede.constraints_utils import breakdown_constr


def test_sample():
    x = cp.Variable((4, 4))
    constr = x.sum(1) <= 1
    out = breakdown_constr(constr, "row")
    print(tester(out))
    expected = [str(x[i, :] <= 1) for i in range(4)]
    assert tester(out) == expected


def test_sample_2():
    x = cp.Variable((4, 4))
    constr = x.sum(0) <= 1
    out = breakdown_constr(constr, "row")
    print(tester(out))
    expected = [str(x[:, i] <= 1) for i in range(4)]
    print(tester(expected))
    assert tester(out) == expected


# def test_failing_sample():
#     x = cp.Variable((4, 4))
#     constr = x.sum(1) == 1
#     out = breakdown_constr(constr)
#     expected = [str(x[i, :] == 1) for i in range(4)]
#     assert tester(out) == expected


# def test_hard1():
#     x = cp.Variable((4, 4))
#     constr = x.sum(1) >= np.arange(4)
#     out = breakdown_constr(constr)
#     expected = [str(x[i, :] >= i) for i in range(4)]
#     assert tester(out) == expected


# def test_extra1():
#     x = cp.Variable()
#     constr = x <= 5
#     out = breakdown_constr(constr)
#     expected = [str(x <= 5)]
#     assert tester(out) == expected


# def test_failing_extra1():
#     x = cp.Variable()
#     constr = x == 5
#     out = breakdown_constr(constr)
#     expected = [str(x == 5)]
#     assert tester(out) == expected


# def test_extra2():
#     x = cp.Variable((4, 4))
#     constr = x[0, :] >= np.arange(4)
#     out = breakdown_constr(constr)
#     expected = [str(x[0, i] >= i) for i in range(4)]
#     assert tester(out) == expected


# def test_failing_extra2():
#     x = cp.Variable((4, 4))
#     constr = x[:, 1] == np.arange(4)
#     out = breakdown_constr(constr)
#     expected = [str(x[i, 1] == i) for i in range(4)]
#     assert tester(out) == expected


# def test_hard3():
#     x = cp.Variable((4, 4))
#     c1 = x.sum(1) >= np.arange(4)
#     c2 = x[0, 0] + x[0, 1] >= 0
#     out = []
#     out.extend(breakdown_constr(c1))
#     out.extend(breakdown_constr(c2))
#     expected = [x[i, :] >= i for i in range(4)] + [x[0, 0] + x[0, 1] >= 0]
#     assert tester(out) == tester(expected)


# def test_column():
#     x = cp.Variable((4, 4))
#     constr = x[:, 0] >= np.arange(4)
#     out = breakdown_constr(constr)
#     expected = [str(x[i, 0] >= i) for i in range(4)]
#     assert tester(out) == expected


# def test_2d_chunk():
#     x = cp.Variable((4, 4))
#     constr = x[:2, 1::2] >= np.arange(4).reshape((2, 2))
#     out = breakdown_constr(constr)
#     expected = [
#         str(x[0, 1] >= 0), str(x[0, 3] >= 1),
#         str(x[1, 1] >= 2), str(x[1, 3] >= 3)
#     ]
#     assert tester(out) == expected


# def test_single_row():
#     x = cp.Variable(4)
#     constr = x[:] >= np.arange(4)
#     out = breakdown_constr(constr)
#     expected = [str(x[i] >= i) for i in range(4)]
#     assert tester(out) == expected


# def test_1d_sum():
#     x = cp.Variable(4)
#     constr = x.sum() >= 1
#     out = breakdown_constr(constr)
#     assert str(out[0]) == str(constr)


# def test_2d_sum():
#     x = cp.Variable((4, 4))
#     constr = x.sum() >= 1
#     out = breakdown_constr(constr)
#     assert str(out[0]) == str(constr)


# def test_sum_vars():
#     x = cp.Variable((4, 4))
#     y = cp.Variable((4, 4))
#     constr = x + y + 1 == 0
#     out = breakdown_constr(constr)
#     expected = [x[tup] + y[tup] + 1 == 0 for tup in np.ndindex(x.shape)]
#     assert tester(out) == tester(expected)


def tester(constr):
    return [str(c) for c in constr]


# if __name__ == '__main__':
#     test_sample()
#     test_sample_2()
#     test_failing_sample()
#     test_hard1()
#     test_extra1()
#     test_failing_extra1()
#     test_extra2()
#     test_failing_extra2()
#     test_hard3()

#     test_column()
#     test_2d_chunk()
#     test_single_row()
#     test_1d_sum()
#     test_2d_sum()
#     test_sum_vars()
