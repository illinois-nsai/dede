import os
import pickle
import time
import itertools
import numpy as np
import cvxpy as cp
import ray

from ..config import TOPOLOGIES_DIR
from ..path_utils import find_paths, graph_copy_with_edge_weights, remove_cycles
from .abstract_formulation import AbstractFormulation, Objective
from .dede_subproblems import SubproblemsWrap

PATHS_DIR = os.path.join(TOPOLOGIES_DIR, "paths", "path-form")


class SubprobCache:
    '''Cache subproblems.'''

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


class DeDeFormulation(AbstractFormulation):
    def __init__(
        self,
        *,
        objective,
        num_paths,
        edge_disjoint=True,
        dist_metric="inv-cap",
        DEBUG=False,
        VERBOSE=False,
        out=None
    ):
        super().__init__(objective, DEBUG, VERBOSE, out)
        if dist_metric != "inv-cap" and dist_metric != "min-hop":
            raise Exception(
                'invalid distance metric: {}; only "inv-cap" and "min-hop" are valid choices'.format(
                    dist_metric
                )
            )
        self._objective = objective
        self._num_paths = num_paths
        self.edge_disjoint = edge_disjoint
        self.dist_metric = dist_metric

        # dede
        self._subprob_cache = SubprobCache()
        self._runtime = 0
        np.random.seed(7)

    @staticmethod
    def paths_full_fname(problem, num_paths, edge_disjoint, dist_metric):
        return os.path.join(
            PATHS_DIR,
            "{}-{}-paths_edge-disjoint-{}_dist-metric-{}-dict.pkl".format(
                problem.name, num_paths, edge_disjoint, dist_metric
            ),
        )

    @staticmethod
    def compute_paths(problem, num_paths, edge_disjoint, dist_metric):
        paths_dict = {}
        G = graph_copy_with_edge_weights(problem.G, dist_metric)
        for s_k in G.nodes:
            for t_k in G.nodes:
                if s_k == t_k:
                    continue
                paths = find_paths(G, s_k, t_k, num_paths, edge_disjoint)
                paths_no_cycles = [remove_cycles(path) for path in paths]
                paths_dict[(s_k, t_k)] = paths_no_cycles
        return paths_dict

    @staticmethod
    def read_paths_from_disk_or_compute(
            problem, num_paths, edge_disjoint, dist_metric):
        paths_fname = DeDeFormulation.paths_full_fname(
            problem, num_paths, edge_disjoint, dist_metric
        )
        print("Loading paths from pickle file", paths_fname)

        try:
            with open(paths_fname, "rb") as f:
                paths_dict = pickle.load(f)
                for key, paths in paths_dict.items():
                    paths_no_cycles = [remove_cycles(path) for path in paths]
                    paths_dict[key] = paths_no_cycles
                print("paths_dict size:", len(paths_dict))
                return paths_dict
        except FileNotFoundError:
            print("Unable to find {}".format(paths_fname))
            paths_dict = DeDeFormulation.compute_paths(
                problem, num_paths, edge_disjoint, dist_metric
            )
            print("Saving paths to pickle file")
            with open(paths_fname, "wb") as w:
                pickle.dump(paths_dict, w)
            return paths_dict

    def get_paths(self, problem):
        if not hasattr(self, "_paths_dict"):
            self._paths_dict = DeDeFormulation.read_paths_from_disk_or_compute(
                problem, self._num_paths, self.edge_disjoint, self.dist_metric
            )
        return self._paths_dict

    ###############################
    # Override superclass methods #
    ###############################

    def solve(self, problem, num_cpus=None, rho=None,
              num_iter=10, num_fix_iter=2, debug=False):
        self._problem = problem

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
            raise ValueError(
                f'{num_cpus} CPUs exceeds upper limit of {os.cpu_count()}.')

        # check whether settings has been changed
        key = self._subprob_cache.make_key(rho, num_cpus)
        if key != self._subprob_cache.key:
            # invalidate old settings
            self._subprob_cache.invalidate()
            self._subprob_cache.key = key
            self._subprob_cache.rho = rho
            # get constraint info
            edge_dict = {}
            self.constrs_gps_r = []
            for idx, (u, v) in enumerate(self.problem.G.edges):
                edge_dict[(u, v)] = idx
                self.constrs_gps_r.append(self.problem.G[u][v]['capacity'])
            paths_dict = self.get_paths(problem)
            self.constrs_gps_d = []
            for src, demands in enumerate(self.problem.traffic_matrix._tm):
                self.constrs_gps_d.append([[], [], []])
                for dst, demand in enumerate(demands):
                    if src != dst:
                        self.constrs_gps_d[-1][0].append(dst)
                        self.constrs_gps_d[-1][1].append(demand)
                        self.constrs_gps_d[-1][2].append([[edge_dict[(u, v)] for u, v in zip(
                            p[:-1], p[1:])] for p in paths_dict[(src, dst)]])
                    else:
                        self.constrs_gps_d[-1][0].append(dst)
                        self.constrs_gps_d[-1][1].append(0)
                        self.constrs_gps_d[-1][2].append([])
            # initialize ray
            ray.shutdown()
            self._subprob_cache.num_cpus = num_cpus
            ray.init(num_cpus=num_cpus)
            # store subproblem in last solution
            self._subprob_cache.probs = self.get_subproblems(num_cpus, rho)
            # get initial demand solutions
            self.sol_d = np.vstack(
                ray.get([prob.get_solution_d.remote() for prob in self._subprob_cache.probs]))
            self.sol_d = self.sol_d[self.param_idx_d_back, :].T

        # update traffic demand values
        ray.get([prob.update_parameters.remote(self.problem.traffic_matrix._tm[param_idx])
                for prob, param_idx in zip(self._subprob_cache.probs, self.param_idx_d)])

        # reset runtime
        self._runtime = 0

        if self._objective == Objective.TOTAL_FLOW:
            for i in range(num_iter):
                if i != 0:
                    start = time.time()

                # resource allocation
                [prob.solve_r.remote(self.sol_d[param_idx], enforce_dpp=True) for prob, param_idx in zip(
                    self._subprob_cache.probs, self.param_idx_r)]
                self.sol_r = np.vstack(
                    ray.get([prob.get_solution_r.remote() for prob in self._subprob_cache.probs]))
                self.sol_r = self.sol_r[self.param_idx_r_back, :].T

                if i == 0:
                    # resource allocate in iter 0 can be done before new demands
                    start = time.time()

                # demand allocation
                [prob.solve_d.remote(self.sol_r[param_idx], enforce_dpp=True) for prob, param_idx in zip(
                    self._subprob_cache.probs, self.param_idx_d)]
                self.sol_d = np.vstack(
                    ray.get([prob.get_solution_d.remote() for prob in self._subprob_cache.probs]))
                self.sol_d = self.sol_d[self.param_idx_d_back, :].T

                stop = time.time()

                self._runtime += stop - start
                obj = self.get_obj()
                r_t, r_process_t, d_t, d_process_t = self.get_t()
                if debug:
                    print(
                        f'Iter {i}: end2end time {(stop - start):.4f} s, naive obj {obj:.4f}')
                    print(f'{r_t.shape[0]} resource subproblems: mean time {r_t[:,0].mean():.2f} ms,',
                          f'mean solve time {r_t[:,1].mean():.2f} ms, mean compilation time {r_t[:,2].mean():.2f} ms,'
                          f'process w/ max time vs mean time {(max(r_process_t)/np.mean(r_process_t)):.2f}')
                    print(f'{d_t.shape[0]} demand subproblems: mean time {d_t[:,0].mean():2f} ms,'
                          f'mean solve time {d_t[:,1].mean():.2f} ms, mean compilation time {d_t[:,2].mean():.2f} ms,'
                          f'process w/ max time vs mean time {(max(d_process_t)/np.mean(d_process_t)):.2f}')

            # fix constraint violation
            assert num_fix_iter > 0
            self.fix_sol_d = self.sol_d
            for i in range(num_fix_iter):
                start = time.time()
                self.fix_sol_r = np.vstack(ray.get([prob.fix_r.remote(
                    self.fix_sol_d[param_idx]) for prob, param_idx in zip(self._subprob_cache.probs, self.param_idx_r)]))
                self.fix_sol_r = self.fix_sol_r[self.param_idx_r_back, :].T
                self.fix_sol_d = np.vstack(ray.get([prob.fix_d.remote(
                    self.fix_sol_r[param_idx]) for prob, param_idx in zip(self._subprob_cache.probs, self.param_idx_d)]))
                self.fix_sol_d = self.fix_sol_d[self.param_idx_d_back, :].T
                stop = time.time()
                self._runtime += stop - start
                obj = self.get_fix_obj()
                if debug:
                    print(
                        f'Fix constraint violation at iter {i}: {(stop - start):.4f} s, obj {obj:.4f}')

        elif self._objective == Objective.MIN_MAX_LINK_UTIL:
            alpha, alpha_lambda_mean = 0, 0
            for i in range(num_iter):
                if i != 0:
                    start = time.time()

                # resource allocation
                self.sol_d = np.hstack(
                    [self.sol_d, alpha * np.ones((self.sol_d.shape[0], 1))])
                [prob.solve_r.remote(self.sol_d[param_idx], enforce_dpp=True, solver=cp.CLARABEL)
                 for prob, param_idx in zip(self._subprob_cache.probs, self.param_idx_r)]
                self.sol_r = np.vstack(
                    ray.get([prob.get_solution_r.remote() for prob in self._subprob_cache.probs]))
                self.sol_r = self.sol_r[self.param_idx_r_back, :].T

                if i == 0:
                    # resource allocate in iter 0 can be done before new demands
                    start = time.time()

                # demand allocation
                [prob.solve_d.remote(self.sol_r[param_idx], enforce_dpp=True) for prob, param_idx in zip(
                    self._subprob_cache.probs, self.param_idx_d)]
                self.sol_d = np.vstack(
                    ray.get([prob.get_solution_d.remote() for prob in self._subprob_cache.probs]))
                self.sol_d = self.sol_d[self.param_idx_d_back, :].T
                # manually solve alpha
                alpha = max(self.sol_r[-1].mean() - alpha_lambda_mean -
                            1 / self._subprob_cache.rho / self.sol_r.shape[1], 0)
                alpha_lambda_mean += alpha - self.sol_r[-1].mean()

                stop = time.time()

                self._runtime += stop - start
                obj = self.get_obj()
                r_t, r_process_t, d_t, d_process_t = self.get_t()
                if debug:
                    print(
                        f'Iter {i}: end2end time {(stop - start):.4f} s, naive obj {obj:.4f}')
                    print(f'{r_t.shape[0]} resource subproblems: mean time {r_t[:,0].mean():.2f} ms,',
                          f'mean solve time {r_t[:,1].mean():.2f} ms, mean compilation time {r_t[:,2].mean():.2f} ms,'
                          f'process w/ max time vs mean time {(max(r_process_t)/np.mean(r_process_t)):.2f}')
                    print(f'{d_t.shape[0]} demand subproblems: mean time {d_t[:,0].mean():2f} ms,'
                          f'mean solve time {d_t[:,1].mean():.2f} ms, mean compilation time {d_t[:,2].mean():.2f} ms,'
                          f'process w/ max time vs mean time {(max(d_process_t)/np.mean(d_process_t)):.2f}')

            # fix constraint violation
            assert num_fix_iter > 0
            self.fix_sol_d = self.sol_d
            for i in range(num_fix_iter):
                start = time.time()
                self.fix_sol_d = np.hstack(
                    [self.sol_d, obj * np.ones((self.sol_d.shape[0], 1))])
                self.fix_sol_r = np.vstack(ray.get([prob.fix_r.remote(
                    self.fix_sol_d[param_idx]) for prob, param_idx in zip(self._subprob_cache.probs, self.param_idx_r)]))
                self.fix_sol_r = self.fix_sol_r[self.param_idx_r_back, :].T
                self.fix_sol_d = np.vstack(ray.get([prob.fix_d.remote(
                    self.fix_sol_r[param_idx]) for prob, param_idx in zip(self._subprob_cache.probs, self.param_idx_d)]))
                self.fix_sol_d = self.fix_sol_d[self.param_idx_d_back, :].T
                stop = time.time()
                self._runtime += stop - start
                obj = self.get_fix_obj()
                if debug:
                    print(
                        f'Fix constraint violation at iter {i}: {(stop - start):.4f} s, obj {obj:.4f}')
        return obj

    def get_subproblems(self, num_cpus, rho):
        # shuffle group order
        constrs_gps_idx_r = np.arange(len(self.constrs_gps_r))
        constrs_gps_idx_d = np.arange(len(self.constrs_gps_d))
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
            # get constraints group
            constrs_r = [self.constrs_gps_r[j] for j in idx_r]
            constrs_d = [self.constrs_gps_d[j] for j in idx_d]

            # build subproblems process
            probs.append(SubproblemsWrap.remote(
                self._objective,
                self.problem.G.number_of_nodes(), self.problem.G.number_of_edges(),
                idx_r, idx_d,
                constrs_r, constrs_d,
                rho))
        self.param_idx_r_back = np.argsort(np.hstack(self.param_idx_r))
        self.param_idx_d_back = np.argsort(np.hstack(self.param_idx_d))
        return probs

    @property
    def sol_mat(self):
        '''return solution matrix [edge, src]'''
        return self.fix_sol_d

    @property
    def runtime(self):
        return self._runtime

    def get_obj(self):
        if self._objective == Objective.TOTAL_FLOW:
            return -sum(ray.get([prob.get_obj.remote()
                        for prob in self._subprob_cache.probs]))
        elif self._objective == Objective.MIN_MAX_LINK_UTIL:
            return max(ray.get([prob.get_obj.remote()
                       for prob in self._subprob_cache.probs]))

    def get_fix_obj(self):
        if self._objective == Objective.TOTAL_FLOW:
            return -sum(ray.get([prob.get_fix_obj.remote()
                        for prob in self._subprob_cache.probs]))
        elif self._objective == Objective.MIN_MAX_LINK_UTIL:
            return max(ray.get([prob.get_fix_obj.remote(self.fix_sol_d[param_idx])
                       for prob, param_idx in zip(self._subprob_cache.probs, self.param_idx_r)]))

    def get_t(self):
        r_t = ray.get([prob.get_r_t.remote()
                      for prob in self._subprob_cache.probs])
        r_process_t = [sum([ts[0] for ts in process_t]) for process_t in r_t]
        r_t = np.vstack(list(itertools.chain.from_iterable(r_t))) * 1000

        d_t = ray.get([prob.get_d_t.remote()
                      for prob in self._subprob_cache.probs])
        d_process_t = [sum([ts[0] for ts in process_t]) for process_t in d_t]
        d_t = np.vstack(list(itertools.chain.from_iterable(d_t))) * 1000

        return r_t, r_process_t, d_t, d_process_t
