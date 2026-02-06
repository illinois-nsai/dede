import itertools
import os
import time

import numpy as np
import ray

from .dede_subproblems import SubproblemsWrap
from .utils import fix


class SubprobCache:
    """Cache subproblems."""

    def __init__(self):
        self.key = None
        self.rho = None
        self.num_cpus = None
        self.probs = None

    def invalidate(self):
        self.key = None
        self.rho = None
        self.num_cpus = None
        self.probs = None

    def make_key(self, rho, num_cpus):
        return (rho, num_cpus)


class DeDeFormulation:
    def __init__(self, M, N, epsilon, memory_limit, search_limit=None):
        self.M = M
        self.N = N

        # dede
        self._subprob_cache = SubprobCache()
        self._epsilon = epsilon
        self._memory_limit = memory_limit
        if search_limit is None:
            self._search_limit = self.N
        else:
            self._search_limit = search_limit
        self._currentLocations = np.zeros((self.M, self.N))
        for shardNum in range(self.N):
            serverNum = shardNum % self.M
            self._currentLocations[serverNum][shardNum] = 1

    def solve(self, shardLoads, num_cpus=None, rho=None, num_iter=10, debug=False):
        self._shardLoads = shardLoads
        self._averageLoad = shardLoads.sum() / self.M

        # initialize num_cpus, rho
        if num_cpus is None:
            if self._subprob_cache.num_cpus is None:
                num_cpus = os.cpu_count()
            else:
                num_cpus = self._subprob_cache.num_cpus
        if rho is None:
            if self._subprob_cache.rho is None:
                rho = 1
            else:
                rho = self._subprob_cache.rho
        # check whether num_cpus is more than all available
        if num_cpus > os.cpu_count():
            raise ValueError(f"{num_cpus} CPUs exceeds upper limit of {os.cpu_count()}.")

        # check whether settings has been changed
        key = self._subprob_cache.make_key(rho, num_cpus)
        if key != self._subprob_cache.key:
            # invalidate old settings
            self._subprob_cache.invalidate()
            self._subprob_cache.key = key
            self._subprob_cache.rho = rho
            # initialize ray
            ray.shutdown()
            self._subprob_cache.num_cpus = num_cpus
            ray.init(num_cpus=num_cpus)
            # store subproblem in last solution
            self._subprob_cache.probs = self.get_subproblems(num_cpus, rho)
            # get initial demand solutions
            self.sol_d = np.vstack(
                ray.get([prob.get_solution_d.remote() for prob in self._subprob_cache.probs])
            )
            self.sol_d = self.sol_d[self.param_idx_d_back, :].T

        # update shards values
        [
            prob.update_parameters.remote(
                self._currentLocations[idx_r], self._averageLoad, self._shardLoads
            )
            for prob, idx_r, idx_d in zip(
                self._subprob_cache.probs, self.param_idx_r, self.param_idx_d
            )
        ]

        self._runtime = 0
        for i in range(num_iter):
            start = time.time()

            # resource allocation
            [
                prob.solve_r.remote(self.sol_d[param_idx], enforce_dpp=True, solver="GUROBI")
                for prob, param_idx in zip(self._subprob_cache.probs, self.param_idx_r)
            ]
            self.sol_r = np.vstack(
                ray.get([prob.get_solution_r.remote() for prob in self._subprob_cache.probs])
            )
            self.sol_r = self.sol_r[self.param_idx_r_back, :].T

            if i == num_iter - 1:
                stop = time.time()

            # demand allocation
            [
                prob.solve_d.remote(self.sol_r[param_idx], enforce_dpp=True)
                for prob, param_idx in zip(self._subprob_cache.probs, self.param_idx_d)
            ]
            self.sol_d = np.vstack(
                ray.get([prob.get_solution_d.remote() for prob in self._subprob_cache.probs])
            )
            self.sol_d = self.sol_d[self.param_idx_d_back, :].T

            if i != num_iter - 1:
                stop = time.time()

            self._runtime += stop - start
            obj = sum(ray.get([prob.get_obj.remote() for prob in self._subprob_cache.probs]))
            r_t, r_process_t, d_t, d_process_t = self.get_t()
            if debug:
                print("iter%d: end2end time %.4f, obj=%.4f" % (i, stop - start, obj))
                print(
                    "%d r %.2f=%.2f+%.2f ms, scheduling overhead %.2f; %d d %.2f=%.2f+%.2f ms, scheduling overhead %.2f"
                    % (
                        r_t.shape[0],
                        r_t.mean(0)[0],
                        r_t.mean(0)[1],
                        r_t.mean(0)[2],
                        max(r_process_t) / np.mean(r_process_t),
                        d_t.shape[0],
                        d_t.mean(0)[0],
                        d_t.mean(0)[1],
                        d_t.mean(0)[2],
                        max(d_process_t) / np.mean(d_process_t),
                    )
                )

        # update currentLocations
        start = time.time()
        self.sol_fix, nextLocations = fix(
            self.sol_r.T,
            self._currentLocations,
            self._averageLoad,
            self._shardLoads,
            self._epsilon,
            self._memory_limit,
        )
        self._runtime += time.time() - start

        obj = self.get_fix_obj(self._currentLocations, nextLocations)
        if debug:
            print(f"obj {obj}, fix time {time.time() - start:.4f}")
        self._currentLocations = nextLocations
        return obj

    def get_subproblems(self, num_cpus, rho):
        # shuffle group order
        constrs_gps_idx_r = np.arange(self.M)
        constrs_gps_idx_d = np.arange(self.N)
        np.random.shuffle(constrs_gps_idx_r)
        np.random.shuffle(constrs_gps_idx_d)

        self.param_idx_r, self.param_idx_d = [], []
        # build actors with subproblems
        probs = []
        for cpu in range(num_cpus):
            # get constraint idx for the group
            idx_r = constrs_gps_idx_r[cpu::num_cpus].tolist()
            idx_d = constrs_gps_idx_d[cpu::num_cpus].tolist()
            self.param_idx_r.append(idx_r)
            self.param_idx_d.append(idx_d)

            # build subproblems process
            probs.append(
                SubproblemsWrap.remote(
                    idx_r,
                    idx_d,
                    self.M,
                    self.N,
                    self._currentLocations[idx_r],
                    self._currentLocations[:, idx_d],
                    self._averageLoad,
                    self._shardLoads,
                    self._epsilon,
                    self._memory_limit,
                    self._search_limit,
                    rho,
                )
            )
        self.param_idx_r_back = np.argsort(np.hstack(self.param_idx_r))
        self.param_idx_d_back = np.argsort(np.hstack(self.param_idx_d))
        return probs

    def get_obj(self):
        return sum(ray.get([prob.get_obj.remote() for prob in self._subprob_cache.probs]))

    def get_fix_obj(self, currentLocations, nextLocations):
        return ((1 - currentLocations) * nextLocations).sum()

    @property
    def runtime(self):
        return self._runtime

    @property
    def sol_mat(self):
        return self.sol_fix

    def get_t(self):
        r_t = ray.get([prob.get_r_t.remote() for prob in self._subprob_cache.probs])
        r_process_t = [sum([ts[0] for ts in process_t]) for process_t in r_t]
        r_t = np.vstack(list(itertools.chain.from_iterable(r_t))) * 1000

        d_t = ray.get([prob.get_d_t.remote() for prob in self._subprob_cache.probs])
        d_process_t = [sum([ts[0] for ts in process_t]) for process_t in d_t]
        d_t = np.vstack(list(itertools.chain.from_iterable(d_t))) * 1000

        return r_t, r_process_t, d_t, d_process_t
