import numpy as np
import cvxpy as cp
from dede.constraints_utils import func


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
    expected = [x[i, :] >= i for i in range(4)] + [x[0, 0] + x[0, 1] >= 0]
    assert tester(out) == tester(expected)


def test_column():
    x = cp.Variable((4, 4))
    constr = x[:, 0] >= np.arange(4)
    out = func(constr)
    expected = [str(x[i, 0] >= i) for i in range(4)]
    assert tester(out) == expected


def test_2d_chunk():
    x = cp.Variable((4, 4))
    constr = x[:2, 1::2] >= np.arange(4).reshape((2, 2))
    out = func(constr)
    expected = [
        str(x[0, 1] >= 0), str(x[0, 3] >= 1),
        str(x[1, 1] >= 2), str(x[1, 3] >= 3)
    ]
    assert tester(out) == expected


def test_single_row():
    x = cp.Variable(4)
    constr = x[:] >= np.arange(4)
    out = func(constr)
    expected = [str(x[i] >= i) for i in range(4)]
    assert tester(out) == expected


def test_1d_sum():
    x = cp.Variable(4)
    constr = x.sum() >= 1
    out = func(constr)
    assert str(out[0]) == str(constr)


def test_2d_sum():
    x = cp.Variable((4, 4))
    constr = x.sum() >= 1
    out = func(constr)
    assert str(out[0]) == str(constr)


def test_sum_vars():
    x = cp.Variable((4, 4))
    y = cp.Variable((4, 4))
    constr = x + y + 1 == 0
    out = func(constr)
    expected = [x[tup] + y[tup] + 1 == 0 for tup in np.ndindex(x.shape)]
    assert tester(out) == tester(expected)


def test_hard2(): #Not fully implemented
    x = cp.Variable((4,4))
    c1 = x[0,2:] + x[2,3] >= 10
    c2 = x[2:] @ np.arange(5,9) >= 0
    out = []
    out.extend(func(c1))
    out.extend(func(c2))
    print("GOT:", tester(out))
    expected = [x[0,2] + x[0,3] + x[2,3] >= 10, x[2,:] @ np.arange(5,9) >= 0, x[3,:] @ np.arange(5,9) >= 0]
    assert tester(out) == tester(expected)


def tester(constr):
    return [str(c) for c in constr]


def test_sum_rows():
    x = cp.Variable((10, 10))
    constr = x.sum(1) <= 5
    out = func(constr)
    expected = [str(x[i, :] <= 5) for i in range(10)]
    assert tester(out) == expected

def test_sum_cols():
    x = cp.Variable((10, 10))
    constr = x.sum(0) <= 3
    out = func(constr)
    expected = [str(x[:, i] <= 3) for i in range(10)]
    assert tester(out) == expected

def test_sum_equal():
    x = cp.Variable((10, 10))
    constr = x.sum(1) == 4
    out = func(constr)
    expected = [str(x[i, :] == 4) for i in range(10)]
    assert tester(out) == expected

def test_vector_rhs():
    x = cp.Variable((10, 10))
    constr = x.sum(1) >= np.arange(10)
    out = func(constr)
    expected = [str(x[i, :] >= i) for i in range(10)]
    assert tester(out) == expected

def test_direct_row():
    x = cp.Variable((10, 10))
    constr = x[3, :] >= np.arange(10)
    out = func(constr)
    expected = [str(x[3, i] >= i) for i in range(10)]
    assert tester(out) == expected

def test_direct_col():
    x = cp.Variable((10, 10))
    constr = x[:, 2] == np.arange(10)
    out = func(constr)
    expected = [str(x[i, 2] == i) for i in range(10)]
    assert tester(out) == expected

def test_chunk():
    x = cp.Variable((10, 10))
    constr = x[:2, 5:] >= np.arange(10).reshape((2, 5))
    out = func(constr)
    expected = [
        str(x[i, j] >= 5 * i + (j - 5)) for i in range(2) for j in range(5, 10)
    ]
    assert tester(out) == expected

def test_single_row_var():
    x = cp.Variable(10)
    constr = x[:] >= np.arange(10)
    out = func(constr)
    expected = [str(x[i] >= i) for i in range(10)]
    assert tester(out) == expected

def test_full_sum():
    x = cp.Variable((10, 10))
    constr = x.sum() >= 5
    out = func(constr)
    assert str(out[0]) == str(constr)

def test_add_expr():
    x = cp.Variable((10, 10))
    y = cp.Variable((10, 10))
    constr = x + y + 2 == 1
    out = func(constr)
    expected = [x[tup] + y[tup] + 2 == 1 for tup in np.ndindex(x.shape)]
    assert tester(out) == tester(expected)





def test_broadcast_scalar_rhs():
    x = cp.Variable((5, 5))
    constr = x + 1 <= 10
    out = func(constr)
    print("GOT:", tester(out))

def test_sum_promote():
    x = cp.Variable((3, 3))
    constr = x.sum(0) + 5 >= 0
    out = func(constr)
    print("GOT:", tester(out))

def test_constant_matrix_rhs():
    x = cp.Variable((3, 3))
    constr = x <= np.ones((3, 3))
    out = func(constr)
    print("GOT:", tester(out))

def test_chain_addition():
    x = cp.Variable((2, 2))
    y = cp.Variable((2, 2))
    z = cp.Variable((2, 2))
    constr = x + y + z == 0
    out = func(constr)
    print("GOT:", tester(out))

def test_multiple_constraints_list():
    x = cp.Variable((3, 3))
    constr = [x[0, :] <= 1, x[1, :] >= 0, x[2, :] == -1]
    out = func(constr)
    print("GOT:", tester(out))

def test_nested_index_and_sum():
    x = cp.Variable((5, 5))
    constr = x[:, 2].sum() >= 5
    out = func(constr)
    print("GOT:", tester(out))




if __name__ == '__main__':
    test_broadcast_scalar_rhs()
    test_sum_promote()
    test_constant_matrix_rhs()
    test_chain_addition()
    test_multiple_constraints_list()
    test_nested_index_and_sum()
#     # test_sample()
#     # test_sample_2()
#     # test_failing_sample()
#     # test_hard1()
#     # test_extra1()
#     # test_failing_extra1()
#     # test_extra2()
#     # test_failing_extra2()
#     # test_hard3()

#     # test_column()
#     # test_2d_chunk()
#     # test_single_row()
#     # #test_hard2()
#     # test_1d_sum()
#     # test_2d_sum()
#     # test_sum_vars()
#     test_sum_rows()
#     test_sum_cols()
#     test_sum_equal()
#     test_vector_rhs()
#     test_direct_row()
#     test_direct_col()
#     test_chunk()
#     test_single_row_var()
#     test_full_sum()
#     test_add_expr()

