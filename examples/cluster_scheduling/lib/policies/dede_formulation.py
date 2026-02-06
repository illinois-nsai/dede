import itertools
import os
import time

import cvxpy as cp
import numpy as np
import ray

from .dede_subproblems import SubproblemsWrap
from .objective import Objective

EPS = 1e-6


class SubprobCache:
    """Cache subproblems."""

    def __init__(self):
        self.key = None
        self.rho = None
        self.num_cpus = None
        self.probs = None
        self.cluster_spec = None
        self.N = 0

    def invalidate(self):
        self.key = None
        self.rho = None
        self.num_cpus = None
        self.probs = None
        self.cluster_spec = None
        self.N = 0

    def make_key(self, rho, num_cpus, cluster_spec, N):
        return (
            rho,
            num_cpus,
            tuple(
                [
                    (worker_type, cluster_spec[worker_type])
                    for worker_type in sorted(list(cluster_spec.keys()))
                ]
            ),
            self.N if N <= self.N else int(1.5 * N),
        )


class DeDeFormulation:
    def __init__(
        self,
        policy,
        num_cpus,
        rho,
        warmup,
        warmup_admm_steps,
        admm_steps,
        fix_steps,
    ):
        self._name = policy._name
        if self._name == "MaxMinFairness_Perf":
            self._objective = Objective.MAX_MIN_ALLOC
        elif self._name == "MaxProportionalFairness":
            self._objective = Objective.TOTAL_UTIL
        else:
            raise ValueError(f"Objective {self._name} is not supported")
        self._solver = policy._solver

        self.num_cpus = num_cpus
        self.rho = rho
        self.warmup = warmup
        self.warmup_admm_steps = warmup_admm_steps
        self.admm_steps = admm_steps
        self.fix_steps = fix_steps

        # dede
        self._subprob_cache = SubprobCache()
        self._runtime = 0

    @property
    def name(self):
        return self._name

    def get_allocation(
        self,
        unflattened_throughputs,
        scale_factors,
        unflattened_priority_weights,
        cluster_spec,
        debug=False,
    ):

        self._job_ids = sorted(list(scale_factors.keys()))
        self._worker_types = sorted(list(cluster_spec.keys()))

        self._num_workers = np.array(
            [cluster_spec[worker_type] for worker_type in self._worker_types]
        )
        self._scale_factors_list = np.array([scale_factors[job_id] for job_id in self._job_ids])
        self._throughputs = np.array(
            [
                [unflattened_throughputs[job_id][worker_type] for job_id in self._job_ids]
                for worker_type in self._worker_types
            ]
        )
        if self._objective == Objective.MAX_MIN_ALLOC:
            priority_weights = np.array(
                [unflattened_priority_weights[job_id] for job_id in self._job_ids]
            )
            self._throughputs = (
                self._throughputs
                / (
                    (self._num_workers / self._num_workers.sum())
                    @ self._throughputs
                    * priority_weights
                    / self._scale_factors_list
                )[None, :]
            )
        elif self._objective == Objective.TOTAL_UTIL:
            self._throughputs = self._throughputs * self._scale_factors_list[None, :]

        # initialize num_cpus, rho
        if self.num_cpus is None:
            if self._subprob_cache.num_cpus is None:
                self.num_cpus = os.cpu_count()
            else:
                self.num_cpus = self._subprob_cache.num_cpus
        if self.rho is None:
            if self._subprob_cache.rho is None:
                self.rho = 1
            else:
                self.rho = self._subprob_cache.rho
        # check whether num_cpus is more than all available
        if self.num_cpus > os.cpu_count():
            raise ValueError(f"{self.num_cpus} CPUs exceeds upper limit of {os.cpu_count()}.")

        # check whether settings has been changed
        key = self._subprob_cache.make_key(
            self.rho, self.num_cpus, cluster_spec, len(scale_factors)
        )
        if key != self._subprob_cache.key:
            # invalidate old settings
            self._subprob_cache.invalidate()
            self._subprob_cache.key = key
            self._subprob_cache.rho = self.rho
            self._subprob_cache.cluster_spec = cluster_spec
            self._subprob_cache.N = key[-1]
            # initial var record
            self.M, self.N = len(cluster_spec), key[-1]
            self.job_id_to_idx_d = {}
            self.vacant_idx_d = {i for i in range(self.N)}
            self.warmup_used = 0
            # initialize ray
            ray.shutdown()
            self._subprob_cache.num_cpus = self.num_cpus
            ray.init(num_cpus=self.num_cpus)
            # store subproblem in last solution
            self._subprob_cache.probs = self.get_subproblems(self.num_cpus, self.rho)
            # get initial demand solutions
            self.sol_d = np.vstack(
                ray.get([prob.get_solution_d.remote() for prob in self._subprob_cache.probs])
            )
            self.sol_d = self.sol_d[self.param_idx_d_back, :].T
            self.alpha, self.alpha_lambda_mean = 0, 0

        for job_id in self._job_ids:
            if job_id not in self.job_id_to_idx_d:
                self.job_id_to_idx_d[job_id] = self.vacant_idx_d.pop()
        for job_id in list(self.job_id_to_idx_d.keys()):
            if job_id not in self._job_ids:
                self.vacant_idx_d.add(self.job_id_to_idx_d.pop(job_id))
        self.valid_idx_d = [self.job_id_to_idx_d[job_id] for job_id in self._job_ids]
        self.is_valid_idx_d = np.zeros(self.N)
        self.is_valid_idx_d[self.valid_idx_d] = 1
        # fit throughput and scale factors into (M, N)
        self._throughputs_ = np.zeros((self.M, self.N))
        self._throughputs_[:, self.valid_idx_d] = self._throughputs
        self._scale_factors_list_ = np.zeros(self.N)
        self._scale_factors_list_[self.valid_idx_d] = self._scale_factors_list

        # update shards values
        [
            prob.update_parameters.remote(
                self._scale_factors_list_,
                self._throughputs_[:, param_idx],
                self.is_valid_idx_d[param_idx],
            )
            for prob, param_idx in zip(self._subprob_cache.probs, self.param_idx_d)
        ]
        self._runtime = 0

        self.warmup_used += 1
        if self._objective == Objective.TOTAL_UTIL:
            for i in range(
                self.warmup_admm_steps if self.warmup_used <= self.warmup else self.admm_steps
            ):
                start = time.time()

                # resource allocation
                # solver=cp.CLARABEL is necessary to avoid solver failure
                [
                    prob.solve_r.remote(self.sol_d[param_idx], enforce_dpp=True, solver=cp.CLARABEL)
                    for prob, param_idx in zip(self._subprob_cache.probs, self.param_idx_r)
                ]
                self.sol_r = np.vstack(
                    ray.get([prob.get_solution_r.remote() for prob in self._subprob_cache.probs])
                )
                self.sol_r = self.sol_r[self.param_idx_r_back, :].T

                # demand allocation
                [
                    prob.solve_d.remote(
                        self.sol_r[param_idx], enforce_dpp=True, solver=self._solver
                    )
                    for prob, param_idx in zip(self._subprob_cache.probs, self.param_idx_d)
                ]
                self.sol_d = np.vstack(
                    ray.get([prob.get_solution_d.remote() for prob in self._subprob_cache.probs])
                )
                self.sol_d = self.sol_d[self.param_idx_d_back, :].T

                stop = time.time()

                self._runtime += stop - start
                obj = self.get_obj()
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

            # fix constraint violation
            assert self.fix_steps > 0
            self.fix_sol_d = self.sol_d
            for i in range(self.fix_steps):
                start = time.time()
                # enlarge r
                self.fix_sol_r = np.vstack(
                    ray.get(
                        [
                            prob.fix_r.remote(self.fix_sol_d[param_idx], i)
                            for prob, param_idx in zip(self._subprob_cache.probs, self.param_idx_r)
                        ]
                    )
                )
                self.fix_sol_r = self.fix_sol_r[self.param_idx_r_back, :].T
                self.fix_sol_d = np.vstack(
                    ray.get(
                        [
                            prob.fix_d.remote(self.fix_sol_r[param_idx], i)
                            for prob, param_idx in zip(self._subprob_cache.probs, self.param_idx_d)
                        ]
                    )
                )
                self.fix_sol_d = self.fix_sol_d[self.param_idx_d_back, :].T
                stop = time.time()
                self._runtime += stop - start
                obj = self.get_fix_obj()
                if debug:
                    print(
                        f"Fix constraint violation at iter {i}: {(stop - start):.4f} s, obj {obj:.4f}"
                    )
        elif self._objective == Objective.MAX_MIN_ALLOC:
            self.sol_d = np.vstack([self.sol_d, self.alpha * np.ones((1, self.sol_d.shape[1]))])
            for i in range(
                self.warmup_admm_steps if self.warmup_used <= self.warmup else self.admm_steps
            ):
                start = time.time()

                # resource allocation
                # solver=cp.CLARABEL is necessary to avoid solver failure
                [
                    prob.solve_r.remote(self.sol_d[param_idx], enforce_dpp=True, solver=cp.CLARABEL)
                    for prob, param_idx in zip(self._subprob_cache.probs, self.param_idx_r)
                ]
                self.sol_r = np.vstack(
                    ray.get([prob.get_solution_r.remote() for prob in self._subprob_cache.probs])
                )
                self.sol_r = self.sol_r[self.param_idx_r_back, :].T
                # manually solve alpha
                sol_d_alpha_valid = self.sol_d[-1, self.valid_idx_d].mean()
                self.alpha_lambda_mean += self.alpha - sol_d_alpha_valid
                self.alpha = max(
                    sol_d_alpha_valid
                    - self.alpha_lambda_mean
                    + 1 / self._subprob_cache.rho / len(self.valid_idx_d),
                    0,
                )

                # demand allocation
                self.sol_r = np.hstack([self.sol_r, self.alpha * np.ones((self.sol_r.shape[0], 1))])
                [
                    prob.solve_d.remote(self.sol_r[param_idx], enforce_dpp=True)
                    for prob, param_idx in zip(self._subprob_cache.probs, self.param_idx_d)
                ]
                self.sol_d = np.vstack(
                    ray.get([prob.get_solution_d.remote() for prob in self._subprob_cache.probs])
                )
                self.sol_d = self.sol_d[self.param_idx_d_back, :].T

                stop = time.time()

                self._runtime += stop - start
                obj = self.get_obj()
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

            # fix constraint violation
            assert self.fix_steps > 0
            self.fix_sol_r = self.sol_d[:-1].T
            obj = self.get_fix_obj()
            for i in range(self.fix_steps):
                start = time.time()
                self.fix_sol_r = np.hstack(
                    [self.fix_sol_r, obj * 1.01 * np.ones((self.sol_r.shape[0], 1))]
                )
                # enlarge d
                self.fix_sol_d = np.vstack(
                    ray.get(
                        [
                            prob.fix_d.remote(self.fix_sol_r[param_idx], i)
                            for prob, param_idx in zip(self._subprob_cache.probs, self.param_idx_d)
                        ]
                    )
                )
                self.fix_sol_d = self.fix_sol_d[self.param_idx_d_back, :].T
                self.fix_sol_r = np.vstack(
                    ray.get(
                        [
                            prob.fix_r.remote(self.fix_sol_d[param_idx], i)
                            for prob, param_idx in zip(self._subprob_cache.probs, self.param_idx_r)
                        ]
                    )
                )
                self.fix_sol_r = self.fix_sol_r[self.param_idx_r_back, :].T
                obj = self.get_fix_obj()
                stop = time.time()
                self._runtime += stop - start
                if debug:
                    print(
                        f"Fix constraint violation at iter {i}: {(stop - start):.4f} s, obj {obj:.4f}"
                    )
            self.fix_sol_d = self.fix_sol_r.T
            for i in range(10):
                start = time.time()
                # enlarge r
                self.fix_sol_r = np.vstack(
                    ray.get(
                        [
                            prob.fix_r.remote(self.fix_sol_d[param_idx], -1)
                            for prob, param_idx in zip(self._subprob_cache.probs, self.param_idx_r)
                        ]
                    )
                )
                self.fix_sol_r = self.fix_sol_r[self.param_idx_r_back, :].T
                self.fix_sol_d = np.vstack(
                    ray.get(
                        [
                            prob.fix_d.remote(self.fix_sol_r[param_idx], -1)
                            for prob, param_idx in zip(self._subprob_cache.probs, self.param_idx_d)
                        ]
                    )
                )
                self.fix_sol_d = self.fix_sol_d[self.param_idx_d_back, :].T
                stop = time.time()
                self.fix_sol_r = self.fix_sol_d.T
                obj = self.get_fix_obj()
                self._runtime += stop - start
                if debug:
                    print(f"After fix at iter {i}: {(stop - start):.4f} s, obj {obj:.4f}")

        var_clip = self.sol_mat
        # print(var_clip.sum(1).max(), (self._scale_factors_list @ var_clip / self._num_workers).max())
        d = {}
        for i, job_id in enumerate(self._job_ids):
            d[job_id] = {
                worker_type: var_clip[i][j] for j, worker_type in enumerate(self._worker_types)
            }
        return d

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
                    self._objective,
                    idx_r,
                    idx_d,
                    self.M,
                    self.N,
                    self._num_workers[idx_r],
                    np.zeros((self.M, len(idx_d))),
                    np.zeros(self.N),
                    rho,
                )
            )
        self.param_idx_r_back = np.argsort(np.hstack(self.param_idx_r))
        self.param_idx_d_back = np.argsort(np.hstack(self.param_idx_d))
        return probs

    def get_obj(self):
        if self._objective == Objective.TOTAL_UTIL:
            return -np.hstack(
                ray.get([prob.get_obj.remote() for prob in self._subprob_cache.probs])
            ).sum()
        elif self._objective == Objective.MAX_MIN_ALLOC:
            return np.hstack(
                ray.get([prob.get_obj.remote() for prob in self._subprob_cache.probs])
            ).min()

    def get_fix_obj(self):
        if self._objective == Objective.TOTAL_UTIL:
            return -np.hstack(
                ray.get([prob.get_fix_obj.remote() for prob in self._subprob_cache.probs])
            ).sum()
        elif self._objective == Objective.MAX_MIN_ALLOC:
            return np.hstack(
                ray.get(
                    [
                        prob.get_fix_obj.remote(self.fix_sol_r[param_idx])
                        for prob, param_idx in zip(self._subprob_cache.probs, self.param_idx_d)
                    ]
                )
            ).min()

    @property
    def runtime(self):
        return self._runtime

    @property
    def sol_mat(self):
        if self._objective == Objective.TOTAL_UTIL:
            return self.fix_sol_d.T[self.valid_idx_d]
        elif self._objective == Objective.MAX_MIN_ALLOC:
            return self.fix_sol_r[self.valid_idx_d]

    def get_t(self):
        r_t = ray.get([prob.get_r_t.remote() for prob in self._subprob_cache.probs])
        r_process_t = [sum([ts[0] for ts in process_t]) for process_t in r_t]
        r_t = np.vstack(list(itertools.chain.from_iterable(r_t))) * 1000

        d_t = ray.get([prob.get_d_t.remote() for prob in self._subprob_cache.probs])
        d_process_t = [sum([ts[0] for ts in process_t]) for process_t in d_t]
        d_t = np.vstack(list(itertools.chain.from_iterable(d_t))) * 1000

        return r_t, r_process_t, d_t, d_process_t
