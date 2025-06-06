import numpy as np
import cvxpy as cp
from constraints_func import func


def test_sample():
    x = cp.Variable((4, 4))
    constr = x.sum(1) <= 1
    out = func(constr)
    expected = [str(x[i, :] <= 1) for i in range(4)]
    assert tester(out) == expected


def test_sample_2():
    x = cp.Variable((4, 4))
    constr = x.sum(0) <= 1
    out = func(constr)
    expected = [str(x[:, i] <= 1) for i in range(4)]
    assert tester(out) == expected


def test_failing_sample():
    x = cp.Variable((4, 4))
    constr = x.sum(1) == 1
    out = func(constr)
    expected = [str(x[i, :] == 1) for i in range(4)]
    assert tester(out) == expected


def test_hard1():
    x = cp.Variable((4, 4))
    constr = x.sum(1) >= np.arange(4)
    out = func(constr)
    expected = [str(x[i, :] >= i) for i in range(4)]
    assert tester(out) == expected


def test_extra1():
    x = cp.Variable()
    constr = x <= 5
    out = func(constr)
    expected = [str(x <= 5)]
    assert tester(out) == expected


def test_failing_extra1():
    x = cp.Variable()
    constr = x == 5
    out = func(constr)
    expected = [str(x == 5)]
    assert tester(out) == expected


def test_extra2():
    x = cp.Variable((4, 4))
    constr = x[0, :] >= np.arange(4)
    out = func(constr)
    expected = [str(x[0, i] >= i) for i in range(4)]
    assert tester(out) == expected


def test_failing_extra2():
    x = cp.Variable((4, 4))
    constr = x[:, 1] == np.arange(4)
    out = func(constr)
    expected = [str(x[i, 1] == i) for i in range(4)]
    assert tester(out) == expected


def test_hard3():
    x = cp.Variable((4, 4))
    c1 = x.sum(1) >= np.arange(4)
    c2 = x[0, 0] + x[0, 1] >= 0
    out = []
    out.extend(func(c1))
    out.extend(func(c2))
    print(tester(out))
    expected = [x[i, :] >= i for i in range(4)] + [x[0, 0] + x[0, 1] >= 0]
    assert tester(out) == tester(expected)


# def test_hard2(): #Not fully implemented
#     x = cp.Variable((4,4))
#     c1 = x[0,2:] + x[2,3] >= 10
#     c2 = x[2:] @ np.arange(5,9) >= 0
#     out = []
#     out.extend(func(c1))
#     out.extend(func(c2))
#     print(tester(out))
#     expected = [x[0,2] + x[0,3] + x[2,3] >= 10, x[2,:] @ np.arange(5,9) >= 0, x[3,:] @ np.arange(5,9) >= 0]
#     assert tester(out) == tester(expected)


def tester(constr):
    return [str(c) for c in constr]
