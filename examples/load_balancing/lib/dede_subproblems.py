import numpy as np
import cvxpy as cp
import time
import ray

from cvxpy.problems.problem import Problem as CpProblem


@ray.remote
class SubproblemsWrap():
    '''Wrap subproblems for one actor in ray.'''

    def __init__(
            self, idx_r, idx_d, M, N,
            currentLocation_rows, currentLocations_cols, averageLoad, shardLoads, epsilon, memory_limit, search_limit,
            rho):
        self.probs_r = []
        self.M, self.N = M, N
        for i in range(len(idx_r)):
            idx, currentLocation = idx_r[i], currentLocation_rows[i]
            self.probs_r.append(
                SubproblemR(
                    (0,
                     idx),
                    self.N,
                    currentLocation,
                    averageLoad,
                    shardLoads,
                    epsilon,
                    memory_limit,
                    search_limit,
                    rho))
        self.probs_d = []
        for i in range(len(idx_d)):
            idx, currentLocation = idx_d[i], currentLocations_cols[:, i]
            self.probs_d.append(SubproblemD((1, idx), self.M, currentLocation, rho))

    def get_solution_r(self):
        '''Get concatenated solution of resource problems.'''
        if self.probs_r:
            return np.vstack([prob.get_solution() for prob in self.probs_r])
        else:
            return np.empty((0, self.N))

    def get_solution_d(self):
        '''Get concatenated solution of demand problems.'''
        if self.probs_d:
            return np.vstack([prob.get_solution() for prob in self.probs_d])
        else:
            return np.empty((0, self.M))

    def solve_r(self, param_values, *args, **kwargs):
        '''Solve resource problems in the current actor sequentially.'''
        aug_lgr = 0
        for prob, param_value in zip(self.probs_r, param_values):
            aug_lgr += prob.solve(param_value, *args, **kwargs)
        return aug_lgr

    def solve_d(self, param_values, *args, **kwargs):
        '''Solve demand problems in the current actor sequentially.'''
        aug_lgr = 0
        for prob, param_value in zip(self.probs_d, param_values):
            aug_lgr += prob.solve(param_value, *args, **kwargs)
        return aug_lgr

    def get_obj(self):
        '''Get the sum of objective values.'''
        obj = 0
        for prob in self.probs_r + self.probs_d:
            obj += prob.get_obj()
        return obj

    def update_parameters(self, currentLocations, averageLoad, shardLoads):
        '''Update parameter value in the current actor.'''
        for prob, param_value in zip(self.probs_r, currentLocations):
            prob.currentLocation = param_value
            prob.averageLoad.value = averageLoad
            prob.shardLoads = shardLoads

    def get_d_t(self):
        t = []
        for prob in self.probs_d:
            t.append(prob.get_t())
        return t

    def get_r_t(self):
        t = []
        for prob in self.probs_r:
            t.append(prob.get_t())
        return t


class SubproblemR(CpProblem):
    def __init__(self, idx, N, currentLocation, averageLoad, shardLoads, epsilon, memory_limit, search_limit, rho):
        self._runtime = 0
        self.id = idx
        self.N = N
        self.rho = rho
        self.epsilon = epsilon
        self.memory_limit = memory_limit
        self.search_limit = search_limit

        self.select_idx = (-currentLocation).argsort(kind='stable')[:self.search_limit]
        # self.var = cp.Variable(N, nonneg=True)
        # self.var.value = currentLocation
        # self.var_ = cp.Variable(N, boolean=True)
        # self.var_.value = currentLocation
        self.var = currentLocation
        self.var_select = cp.Variable(self.search_limit, nonneg=True)
        self.var_select.value = currentLocation[self.select_idx]
        self.var_select_ = cp.Variable(self.search_limit, boolean=True)
        self.var_select_.value = currentLocation[self.select_idx]
        self.s = cp.Variable(2, nonneg=True)
        self.s.value = np.zeros(self.s.shape)
        # self.currentLocation = cp.Parameter(N, value=currentLocation)
        self.currentLocation = currentLocation
        self.currentLocation_select = cp.Parameter(self.search_limit, value=currentLocation[self.select_idx])
        # self.shardLoads = cp.Parameter(N, value=shardLoads)
        self.shardLoads = shardLoads
        self.shardLoads_select = cp.Parameter(self.search_limit, value=shardLoads[self.select_idx])
        self.averageLoad = cp.Parameter(value=averageLoad)
        # self.param = cp.Parameter(N, value=currentLocation)
        self.param_select = cp.Parameter(self.search_limit, value=currentLocation[self.select_idx])

        self.f1 = cp.hstack([
            self.var_select @ self.shardLoads_select + self.s[0] - self.averageLoad * (1 + self.epsilon),
            self.var_select @ self.shardLoads_select - self.s[1] - self.averageLoad * (1 - self.epsilon),
        ])
        self.l1 = cp.Parameter(self.f1.shape, value=np.zeros(self.f1.shape))
        # change l2 and f2 if we need to solve under-allocation of large loads
        self.f2 = self.var_select - self.param_select
        self.l2 = np.zeros(N)
        self.l2_select = cp.Parameter(self.f2.shape, value=np.zeros(self.f2.shape))

        super(SubproblemR, self).__init__(
            cp.Minimize(
                (1 - self.currentLocation_select) @ self.var_select_ +
                self.rho / 2 * cp.sum_squares(self.f1 + self.l1) +
                self.rho / 2 * cp.sum_squares(self.f2 + self.l2_select)),
            [self.var_select <= self.var_select_, self.var_select_.sum() <= 16]
        )

    def get_solution(self):
        return self.var

    def get_obj(self):
        return (1 - self.currentLocation_select.value) @ self.var_select_.value

    def solve(self, param_value, *args, **kwargs):
        start = time.time()
        self.select_idx = (-param_value).argsort(kind='stable')[:self.search_limit]
        self.shardLoads_select.value = self.shardLoads[self.select_idx]
        self.currentLocation_select.value = self.currentLocation[self.select_idx]

        self.l1.value += self.f1.value
        # self.param.value = param_value
        self.param_select.value = param_value[self.select_idx]
        # self.l2.value += self.f2.value
        self.l2 += self.var - param_value
        self.l2_select.value = self.l2[self.select_idx]
        res = super(SubproblemR, self).solve(*args, **kwargs)

        self.var = np.zeros(self.N)
        self.var[self.select_idx] = self.var_select.value
        self._runtime = time.time() - start
        return res

    def get_t(self):
        return [self._runtime, self.solver_stats.solve_time, self.compilation_time]


class SubproblemD(CpProblem):
    def __init__(self, idx, M, currentLocation, rho):
        self._runtime = 0
        self.id = idx
        self.M = M
        self.rho = rho

        self.var = cp.Variable(M, nonneg=True)
        self.var.value = currentLocation
        self.param = cp.Parameter(M)
        self.param.value = currentLocation

        self.f1 = self.var.sum() - 1
        self.l1 = cp.Parameter(value=0)
        self.f2 = self.var - self.param
        self.l2 = cp.Parameter(self.f2.shape, value=np.zeros(self.f2.shape))

        super(SubproblemD, self).__init__(
            cp.Minimize(
                cp.sum_squares(self.f1 + self.l1) +
                cp.sum_squares(self.f2 + self.l2)),
        )

    def get_solution(self):
        return self.var.value

    def get_obj(self):
        return 0

    def solve(self, param_value, *args, **kwargs):
        start = time.time()
        self.l1.value += self.f1.value
        # note: we haven't update param yet
        self.l2.value += self.f2.value
        self.param.value = param_value
        res = super(SubproblemD, self).solve(*args, **kwargs)
        self._runtime = time.time() - start
        return res

    def get_t(self):
        return [self._runtime, self.solver_stats.solve_time, self.compilation_time]
