import contextlib
import functools
import time
import typing as t
from collections import defaultdict

import cvxpy as cp
import numpy as np
import ray
import ray.actor
from cvxpy.constraints.nonpos import Inequality
from cvxpy.constraints.zero import Equality, Zero
from cvxpy.problems.objective import Minimize
from cvxpy.problems.problem import Problem as CpProblem
from numpy.typing import NDArray
from ray.util.placement_group import PlacementGroup
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from .constraints_utils import breakdown_constr
from .subproblems_wrap import SubproblemsWrap
from .utils import (
    UnionFind,
    VarInfoT,
    expand_expr,
    get_var_id_pos_list_from_linear,
    get_var_id_pos_list_from_tree,
)

ObjectiveT = t.Union[cp.Maximize, cp.Minimize]
ConstraintT = t.Union[Equality, Zero, Inequality]

# These define parameters needed to be passed to the solvers to make them only use one thread.
THREAD_OPTS: dict[str, dict[str, int]] = {
    cp.GUROBI: {"Threads": 1},
    # ECOS/ECOS_BB do not have settings for threading
    cp.ECOS: {},
    cp.ECOS_BB: {},
}


def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        print(f"Executed {func.__name__} in {end_time - start_time:.4f}s")
        return result

    return wrapper


def time_all_methods(cls):
    for name, val in vars(cls).items():
        if callable(val):
            setattr(cls, name, timer(val))
    return cls


@contextlib.contextmanager
def _get_distributed_pg(
    max_cpus_per_node: int, timeout: float = 10.0
) -> t.Iterator[PlacementGroup]:
    """Returns a placement group that tries to spread
    workers across all available nodes in the ray network.

    Ideally used as a context manager to free up
    the placement group once execution has finished.

    Args:
        max_cpus_per_node (int): how many CPUs to request per node. This is an upper bound on the
        number of CPUs reserved.

    Returns:
        t.Iterator[PlacementGroup]: the placement group
    """
    nodes = [node for node in ray.nodes() if node["Alive"]]

    bundles = [{"CPU": 1.0}] * min(max_cpus_per_node * len(nodes), _get_ray_cpus())
    pg = ray.util.placement_group(bundles, strategy="SPREAD")
    ray.get(pg.ready(), timeout=timeout)
    try:
        yield pg
    finally:
        ray.util.remove_placement_group(pg)


def _get_ray_cpus() -> int:
    """Get the true amount of cpus available in the ray cluster."""
    if not ray.is_initialized():
        raise RuntimeError("Ray is not initialized. Cannot get number of CPUs in the cluster.")
    return int(ray.cluster_resources().get("CPU", 1))


class RaySubprobCache:
    """Cache subproblems."""

    def __init__(self):
        # these three fields reflect user-passed arguments
        self._rho: t.Optional[float] = None
        # this does not necessarily reflect the true number of cpus
        # in the ray cluster if the user inputted None (i.e., use max)
        self._user_num_cpus: t.Optional[int] = None
        self._address: t.Optional[str] = None

        self._probs: t.Optional[list[ray.actor.ActorProxy[SubproblemsWrap]]] = None
        self._param_idx_r: list[list[int]] = []
        self._param_idx_d: list[list[int]] = []
        self._placement_group: t.Optional[PlacementGroup] = None

    def update_cache(self, rho: float, user_num_cpus: t.Optional[int], ray_address: str) -> bool:
        """Updates the execution parameters stored in the cache.

        If this results in cache invalidation, the subproblem settings need to be set again.

        Args:
            rho (float): the rho to use in optimization
            user_num_cpus (t.Optional[int]): how many CPUs to use in optimization. None means all
                available CPUs will be used.
            ray_address (str): the address of the ray server to connect to.
                Either "local" or an ip address.

        Returns:
            bool: True if the cache was invalidated as part of the update
            (i.e., there was a change). Otherwise, False.
        """
        # we want to rebuild subproblems and restart ray if
        # the rho changes, the address or changes, or the num cpus changes in local mode
        if not (
            rho != self._rho
            or ray_address != self._address
            or (user_num_cpus != self._user_num_cpus)
            or not ray.is_initialized()
        ):
            return False

        self._invalidate()
        self._rho = rho
        self._user_num_cpus = user_num_cpus
        self._address = ray_address

        ray.shutdown()
        if ray_address != "local":
            ray.init(address=ray_address)
        else:
            ray.init(num_cpus=user_num_cpus)

        if user_num_cpus is not None and user_num_cpus > _get_ray_cpus():
            raise ValueError(f"Too many CPUs requested, only have {_get_ray_cpus()} available")

        return True

    def reserve_placement_group(self, timeout: float = 10.0) -> PlacementGroup:
        if self._placement_group is not None:
            raise RuntimeError("Reserving another placement group when one is already reserved")
        self._placement_group = ray.util.placement_group([{"CPU": 1}] * (self.num_cpus))
        ray.get(self._placement_group.ready(), timeout=timeout)
        return self._placement_group

    def set_subprobs(
        self,
        probs: list[ray.actor.ActorProxy[SubproblemsWrap]],
        param_idx_r: list[list[int]],
        param_idx_d: list[list[int]],
    ) -> None:
        """Stores subproblems in the cache (subproblems, indices of parameters)."""
        self._probs = probs
        self._param_idx_r = param_idx_r
        self._param_idx_d = param_idx_d

    def _invalidate(self):
        self._rho = None
        self._user_num_cpus = None
        self._address = None
        self._probs = None
        self._param_idx_r, self._param_idx_d = [], []
        if self._placement_group is not None and ray.is_initialized():
            ray.util.remove_placement_group(self._placement_group)
        self._placement_group = None

    @property
    def rho(self) -> float:
        if self._rho is None:
            raise RuntimeError("rho is not set")
        return self._rho

    @property
    def num_cpus(self) -> int:
        """Get the number of CPUs the user requested.
        In the case of None, defauls to all available CPUs."""
        if self._user_num_cpus is None:
            return _get_ray_cpus()
        return self._user_num_cpus

    @property
    def address(self) -> str:
        if self._address is None:
            raise RuntimeError("address is not set")
        return self._address

    @property
    def probs(self) -> list[ray.actor.ActorProxy[SubproblemsWrap]]:
        if self._probs is None:
            raise RuntimeError("probs is not set")
        return self._probs

    @property
    def param_idx_r(self) -> list[list[int]]:
        if not self._param_idx_r:
            raise RuntimeError("param_idx_r is not set")
        return self._param_idx_r

    @property
    def param_idx_d(self) -> list[list[int]]:
        if not self._param_idx_d:
            raise RuntimeError("param_idx_d is not set")
        return self._param_idx_d

    @property
    def placement_group(self) -> PlacementGroup:
        """Get a placement group with the requested number of CPUs."""
        if self._placement_group is None:
            raise RuntimeError("placement group is not set")
        return self._placement_group


@time_all_methods
class Problem(CpProblem):
    """Build a resource allocation problem."""

    def __init__(
        self,
        objective: ObjectiveT,
        resource_constraints: list[ConstraintT],
        demand_constraints: list[ConstraintT],
    ):
        """Initialize problem with the objective and constraints.
        Args:
            objective: Minimize or Maximize. The problem's objective
            resource_variables: list of resource constraints
            demand_variables: list of demand constraints
        """
        # breakdown constraints
        constrs_r_converted = [self._convert_inequality(constr) for constr in resource_constraints]
        constrs_d_converted = [self._convert_inequality(constr) for constr in demand_constraints]

        self._constrs_r = breakdown_constr(constrs_r_converted, 0)
        self._constrs_d = breakdown_constr(constrs_d_converted, 1)

        # init subprob cache
        self._subprob_cache = RaySubprobCache()

        # keep track of original problem type
        self._problem_type = type(objective)

        # choose solver for depending on LP or ILP/MILP
        has_int = False
        for constr in resource_constraints + demand_constraints:
            for v in constr.variables():
                if v.attributes.get("integer", False) or v.attributes.get("boolean", False):
                    has_int = True

        for v in objective.variables():
            if v.attributes.get("integer", False) or v.attributes.get("boolean", False):
                has_int = True

        self._solver = cp.ECOS_BB if has_int else cp.ECOS

        # Initialize original problem
        super(Problem, self).__init__(
            objective if self._problem_type == Minimize else Minimize(-objective.expr),
            self._constrs_r + self._constrs_d,
        )

        # get a dict mapping from param_id to value
        params = t.cast(list[cp.Parameter], self.parameters())
        self.param_id_to_param = {t.cast(int, param.id): param for param in params}

        # get a dict mapping from constraints to list of (var_id, position)
        self.constr_dict_r = self._get_constr_dict(self._constrs_r)
        self.constr_dict_d = self._get_constr_dict(self._constrs_d)

        # get constraints groups
        self.constrs_gps_r = self._group_constrs(self._constrs_r, self.constr_dict_r)
        self.constrs_gps_d = self._group_constrs(self._constrs_d, self.constr_dict_d)

        self._obj_expr_r = None
        self._obj_expr_d = None

    @classmethod
    def _convert_inequality(cls, constr: ConstraintT) -> cp.Constraint:
        if isinstance(constr, Zero) or isinstance(constr, Equality):
            return constr
        elif isinstance(constr, Inequality):
            return constr.expr + cp.Variable(constr.shape, nonneg=True) == 0
        else:
            raise ValueError(f"Constraint {constr} is neither equality nor inequality.")

    def solve(
        self,
        num_cpus: t.Optional[int] = None,
        ray_address: str = "local",
        enable_dede: bool = True,
        rho: float = 1.0,
        num_iter: t.Optional[int] = None,
        xi: t.Optional[float] = None,
        mu: t.Optional[float] = None,
        balance_iterations: t.Optional[int] = None,
        *args,
        **kwargs,
    ) -> np.floating[t.Any]:
        """Compiles and solves the original problem.

        Args:
            num_cpus: number of CPUs to use; if None, use all available CPUs.
                If more than the available number of CPUs, will error.
            ray_address: ray cluster address; if None, will use local ray instance.
            enable_dede: whether to decouple and decompose with DeDe
            num_cpus: number of CPUs to use; all the CPUs available if None
            rho: rho value in ADMM; 1 by default
            num_iter: ADMM iterations; stop under residual tolerances if None
            xi: normalized residual balancing scale; 0.1 by default
            mu: residual imbalance threshold; 10 by default
            balance_iterations: residual balancing frequency; 10 by default
        """
        # solve the original problem
        if not enable_dede:
            start = time.time()
            super(Problem, self).solve(*args, **kwargs)
            end = time.time()
            self._total_time = end - start

            coeff = 1 if self._problem_type == Minimize else -1
            return t.cast(np.floating[t.Any], coeff * self.value)

        kwargs.update(THREAD_OPTS.get(kwargs.get("solver", ""), {}))

        subproblems_invalidated = self._subprob_cache.update_cache(rho, num_cpus, ray_address)
        if subproblems_invalidated:
            # store subproblem in last solution
            obj_expr_r, obj_expr_d = self._get_grouped_objectives()
            # reserve the actors needed to create the subproblems
            self._subprob_cache.reserve_placement_group()
            probs = self.get_subproblems(obj_expr_r, obj_expr_d, self._subprob_cache.num_cpus, rho)

            # store parameter index in z solutions for x problems
            param_idx_r, param_idx_d = self._get_param_idx(probs)

            self._subprob_cache.set_subprobs(probs, param_idx_r, param_idx_d)

            # get demand solution
            self.sol_d = np.hstack(ray.get([prob.get_solution_d.remote() for prob in probs]))

        # update parameter values
        param_id_to_value = {
            param_id: t.cast(NDArray[np.floating[t.Any]], param.value)
            for param_id, param in self.param_id_to_param.items()
        }
        ray.get(
            [prob.update_parameters.remote(param_id_to_value) for prob in self._subprob_cache.probs]
        )

        # solve problem
        # use num_iter if specified
        # otherwise, stop under residual tolerances or reach 10000 upper limit
        i = 0

        mu = 10 if mu is None else mu
        xi = 0.1 if xi is None else xi
        balance_iterations = 10 if balance_iterations is None else balance_iterations
        max_tau = 200
        min_rho = 0.05
        max_rho = 100

        if xi <= 0 or mu <= 0:
            raise ValueError("xi and mu must be positive.")
        if balance_iterations < 1:
            raise ValueError("balance_iterations must be at least 1.")

        self.sol_d_old = self.sol_d.copy()
        self.scaled_dual: dict[VarInfoT, float] = {}

        start = time.time()
        terminate_flag = False
        while (num_iter is not None and i < num_iter) or (num_iter is None and i < 10000):
            if i > 0 and i % balance_iterations == 0:
                primal_res, dual_res, eps_primal, eps_dual = (
                    self.get_relative_residuals_and_epsilon()
                )
                rho_update = "hold"

                if num_iter is None and primal_res <= eps_primal and dual_res <= eps_dual:
                    if not terminate_flag:
                        terminate_flag = True
                    else:
                        break
                else:
                    terminate_flag = False

                if not terminate_flag:
                    tau = max_tau
                    ratio = np.inf
                    if dual_res > 0:
                        ratio = np.sqrt((1 / xi) * primal_res / dual_res)
                    if primal_res == 0 and dual_res == 0:
                        ratio = 1
                    if 1 <= ratio < max_tau:
                        tau = ratio
                    elif 1 / max_tau < ratio < 1:
                        tau = np.sqrt(xi * dual_res / primal_res)

                    if primal_res > xi * mu * dual_res:
                        rho *= tau
                        if rho >= max_rho:
                            rho = max_rho
                            print("Maximum rho reached. Consider adjusting xi up.")

                        for prob in self._subprob_cache.probs:
                            prob.update_rho.remote(rho)
                        rho_update = f"up x{tau:.3e}"
                    elif dual_res > (1 / xi) * mu * primal_res:
                        rho /= tau
                        if rho <= min_rho:
                            rho = min_rho
                            print("Minimum rho reached. Consider adjusting xi down")

                        for prob in self._subprob_cache.probs:
                            prob.update_rho.remote(rho)
                        rho_update = f"down /{tau:.3e}"

                print(
                    f"iter {i}: "
                    f"primal {primal_res:.3e}/{eps_primal:.3e}, "
                    f"dual {dual_res:.3e}/{eps_dual:.3e}, "
                    f"rho {rho:.3e}, "
                    f"update {rho_update}, "
                    f"terminate_flag={terminate_flag}"
                )

            self.sol_d_old = self.sol_d.copy()
            i += 1

            # resource allocation
            ray.get(
                [
                    prob.solve_r.remote(self.sol_d[param_idx], *args, **kwargs)
                    for prob, param_idx in zip(
                        self._subprob_cache.probs, self._subprob_cache.param_idx_r
                    )
                ]
            )
            self.sol_r: NDArray[np.floating[t.Any]] = np.hstack(
                ray.get([prob.get_solution_r.remote() for prob in self._subprob_cache.probs])
            )

            # demand allocation
            ray.get(
                [
                    prob.solve_d.remote(self.sol_r[param_idx], *args, **kwargs)
                    for prob, param_idx in zip(
                        self._subprob_cache.probs, self._subprob_cache.param_idx_d
                    )
                ]
            )
            self.sol_d: NDArray[np.floating[t.Any]] = np.hstack(
                ray.get([prob.get_solution_d.remote() for prob in self._subprob_cache.probs])
            )

        end = time.time()
        print("DeDe Solve Time:", end - start)

        self.populate_vars_with_solution()
        coeff = 1 if self._problem_type == Minimize else -1
        return t.cast(
            np.floating[t.Any],
            coeff * sum(ray.get([prob.get_obj.remote() for prob in self._subprob_cache.probs])),
        )

    def get_relative_residuals_and_epsilon(
        self,
    ) -> tuple[float, float, float, float]:
        """Compute residuals and corresponding primal/dual epsilons."""
        assert self._subprob_cache.probs is not None

        sol_idx_d = ray.get(
            [prob.get_solution_idx_d.remote() for prob in self._subprob_cache.probs]
        )
        sol_idx_r = ray.get(
            [prob.get_solution_idx_r.remote() for prob in self._subprob_cache.probs]
        )
        flat_idx_d = [idx for arr in sol_idx_d for idx in arr]
        flat_idx_r = [idx for arr in sol_idx_r for idx in arr]

        map_r = {k: float(v) for k, v in zip(flat_idx_r, self.sol_r)}
        map_d = {k: float(v) for k, v in zip(flat_idx_d, self.sol_d)}
        map_d_old = {k: float(v) for k, v in zip(flat_idx_d, self.sol_d_old)}

        shared_pos = sorted(set(map_r.keys()) & set(map_d.keys()))
        for pos in shared_pos:
            self.scaled_dual[pos] = self.scaled_dual.get(pos, 0) + map_r[pos] - map_d[pos]

        shared_r = np.array([map_r[pos] for pos in shared_pos])
        shared_d = np.array([map_d[pos] for pos in shared_pos])
        primal_num = np.linalg.norm(shared_r - shared_d)
        primal_denom = max(np.linalg.norm(shared_r), np.linalg.norm(shared_d))

        shared_d_old = np.array([map_d_old[pos] for pos in shared_pos])
        scaled_dual_arr = np.array(list(self.scaled_dual.values()))
        dual_num = np.linalg.norm(shared_d - shared_d_old)
        dual_denom = np.linalg.norm(scaled_dual_arr)

        if primal_denom == 0:
            primal_res = 0 if primal_num == 0 else np.inf
        else:
            primal_res = float(primal_num / primal_denom)

        if dual_denom == 0:
            dual_res = 0 if dual_num == 0 else np.inf
        else:
            dual_res = float(dual_num / dual_denom)

        eps_abs = 0.005
        eps_rel = 0.005
        x_dim = len(shared_pos)

        eps_primal = (
            np.inf
            if primal_denom == 0
            else float(np.sqrt(x_dim) * eps_abs / primal_denom + eps_rel)
        )
        eps_dual = (
            np.inf if dual_denom == 0 else float(np.sqrt(x_dim) * eps_abs / dual_denom + eps_rel)
        )

        return primal_res, dual_res, eps_primal, eps_dual

    def populate_vars_with_solution(self) -> None:
        """Fills problem variables with computed solutions."""
        var_id_to_var = {var.id: var for var in self.variables()}
        for var in self.variables():
            var.value = np.zeros(var.shape)

        local_sol_idx: list[list[VarInfoT]] = ray.get(
            [prob.get_local_solution_idx.remote() for prob in self._subprob_cache.probs]
        )
        local_sol: list[list[np.floating[t.Any]]] = ray.get(
            [prob.get_local_solution.remote() for prob in self._subprob_cache.probs]
        )
        flat_local_idx = [idx for arr in local_sol_idx for idx in arr]
        flat_local_sol: list[np.float64] = [sol for arr in local_sol for sol in arr]

        sol_idx_d: list[list[VarInfoT]] = ray.get(
            [prob.get_solution_idx_d.remote() for prob in self._subprob_cache.probs]
        )
        sol_idx_r: list[list[VarInfoT]] = ray.get(
            [prob.get_solution_idx_r.remote() for prob in self._subprob_cache.probs]
        )
        flat_idx_d = [idx for arr in sol_idx_d for idx in arr]
        flat_idx_r = [idx for arr in sol_idx_r for idx in arr]

        for sol_idx, sol in zip(
            [flat_local_idx, flat_idx_d, flat_idx_r], [flat_local_sol, self.sol_d, self.sol_r]
        ):
            for (var_id, pos), value in zip(sol_idx, sol):
                var = var_id_to_var[var_id]
                idx = np.unravel_index(pos, var.shape[::-1])[::-1]
                var.value[idx] = value

    @classmethod
    def _get_constr_dict(cls, constrs: list[cp.Constraint]) -> dict[int, list[VarInfoT]]:
        """Get a mapping of constraint to its var_id_pos_list."""
        constr_to_var_id_pos_list: dict[int, list[VarInfoT]] = {}
        for constr in constrs:
            # [constr] = get_var_id_pos_list_from_linear(constr.expr, self._solver)
            constr_to_var_id_pos_list[t.cast(int, constr.id)] = get_var_id_pos_list_from_linear(
                constr.expr
            )
        return constr_to_var_id_pos_list

    @classmethod
    def _group_constrs(
        cls, constrs: list[cp.Constraint], constr_dict: dict[int, list[VarInfoT]]
    ) -> list[list[cp.Constraint]]:
        """Group constraints into non-overlapped groups with union-find."""
        uf = UnionFind(len(constrs))

        var_id_pos_to_i: dict[VarInfoT, int] = {}
        for i, constr in enumerate(constrs):
            for var_id_pos in constr_dict[constr.id]:
                if var_id_pos in var_id_pos_to_i:
                    uf.union(var_id_pos_to_i[var_id_pos], i)
                var_id_pos_to_i[var_id_pos] = i

        parent_to_constrs: dict[int, list[cp.Constraint]] = defaultdict(list)
        for i in range(len(constrs)):
            parent_to_constrs[uf.find(i)].append(constrs[i])
        return list(parent_to_constrs.values())

    def get_subproblems(
        self,
        obj_expr_r: list[cp.Expression],
        obj_expr_d: list[cp.Expression],
        num_cpus: int,
        rho: float,
    ) -> list[ray.actor.ActorProxy[SubproblemsWrap]]:
        """Return objective and constraints assignments for subproblems."""

        # shuffle group order
        constrs_gps_idx_r = np.arange(len(self.constrs_gps_r))
        constrs_gps_idx_d = np.arange(len(self.constrs_gps_d))

        np.random.shuffle(constrs_gps_idx_r)  # noqa: NPY002 TODO: replace with np.random.Generator at some point
        np.random.shuffle(constrs_gps_idx_d)  # noqa: NPY002 TODO: replace with np.random.Generator at some point

        # get the set of var_id_pos
        var_id_pos_set_r = set[VarInfoT]()
        for var_id_pos in self.constr_dict_r.values():
            var_id_pos_set_r.update(var_id_pos)
        var_id_pos_set_d = set[VarInfoT]()
        for var_id_pos in self.constr_dict_d.values():
            var_id_pos_set_d.update(var_id_pos)

        # serialize expensive objects exactly once
        obj_expr_r_ref = ray.put(obj_expr_r)
        obj_expr_d_ref = ray.put(obj_expr_d)
        constrs_r_ref = ray.put(self.constrs_gps_r)
        constrs_d_ref = ray.put(self.constrs_gps_d)
        constr_dict_r_ref = ray.put(self.constr_dict_r)
        constr_dict_d_ref = ray.put(self.constr_dict_d)
        var_id_pos_set_r_ref = ray.put(var_id_pos_set_r)
        var_id_pos_set_d_ref = ray.put(var_id_pos_set_d)

        # build actors with subproblems
        probs: list[ray.actor.ActorProxy[SubproblemsWrap]] = []
        for cpu in range(num_cpus):
            # get constraint idx for the group
            idx_r: NDArray[np.signedinteger] = constrs_gps_idx_r[cpu::num_cpus]
            idx_d: NDArray[np.signedinteger] = constrs_gps_idx_d[cpu::num_cpus]

            # build subproblems
            actor = ray.remote(SubproblemsWrap).options(
                scheduling_strategy=PlacementGroupSchedulingStrategy(
                    placement_group=self._subprob_cache.placement_group,
                    placement_group_bundle_index=cpu,
                )
            )
            probs.append(
                actor.remote(
                    idx_r,
                    idx_d,
                    obj_expr_r_ref,
                    obj_expr_d_ref,
                    constrs_r_ref,
                    constrs_d_ref,
                    constr_dict_r_ref,
                    constr_dict_d_ref,
                    var_id_pos_set_r_ref,
                    var_id_pos_set_d_ref,
                    rho,
                )
            )
        return probs

    @classmethod
    def _get_param_idx(
        cls, probs: list[ray.actor.ActorProxy[SubproblemsWrap]]
    ) -> tuple[list[list[int]], list[list[int]]]:
        """Get parameter z index in last solution."""
        # map var_id_pos in the big resource solution list
        sol_idx_r_futures = [prob.get_solution_idx_r.remote() for prob in probs]
        sol_idx_l_futures = [prob.get_solution_idx_d.remote() for prob in probs]
        sol_idx_r: list[list[VarInfoT]] = ray.get(sol_idx_r_futures)

        sol_idx_dict_r: dict[VarInfoT, int] = {}
        idx = 0
        for sol_idx in sol_idx_r:
            for var_id_pos in sol_idx:
                sol_idx_dict_r[var_id_pos] = idx
                idx += 1

        # map var_id_pos in the big demand solution list
        sol_idx_d: list[list[VarInfoT]] = ray.get(sol_idx_l_futures)
        sol_idx_dict_d: dict[VarInfoT, int] = {}
        idx = 0
        for sol_idx in sol_idx_d:
            for var_id_pos in sol_idx:
                sol_idx_dict_d[var_id_pos] = idx
                idx += 1

        # get parameter index
        param_idx_r = [
            [sol_idx_dict_d[var_id_pos] for var_id_pos in sol_idx] for sol_idx in sol_idx_r
        ]
        param_idx_d = [
            [sol_idx_dict_r[var_id_pos] for var_id_pos in sol_idx] for sol_idx in sol_idx_d
        ]
        return param_idx_r, param_idx_d

    def _get_grouped_objectives(self) -> tuple[list[cp.Expression], list[cp.Expression]]:
        """Split objective into corresponding constraint groups"""

        # See if the grouped objectives are already computed and cached.
        # If so, return them to avoid redundant computation.
        if self._obj_expr_r is not None and self._obj_expr_d is not None:
            return self._obj_expr_r, self._obj_expr_d

        # use a placement group with strategy = SPREAD due to the diminishing returns
        # observed with larger numbers of CPUs
        with _get_distributed_pg(4) as pg:
            var_id_pos_to_idx: dict[VarInfoT, list[tuple[int, int]]] = defaultdict(list)
            for i, constrs_gps, constr_dict in zip(
                [0, 1],
                [self.constrs_gps_r, self.constrs_gps_d],
                [self.constr_dict_r, self.constr_dict_d],
            ):
                for j, constrs in enumerate(constrs_gps):
                    for constr in constrs:
                        for var_id_pos in constr_dict[constr.id]:
                            var_id_pos_to_idx[var_id_pos].append((i, j))

            expr_list = expand_expr(self.objective.expr)

            # put heavy objects in shared memory to avoid serialization overhead
            expr_ref = ray.put(expr_list)
            dict_ref = ray.put(dict(var_id_pos_to_idx))

            # chunk the indices to split the work
            chunks = np.array_split(np.arange(len(expr_list), dtype=np.int64), len(pg.bundle_specs))

            # send the chunks to the remote function for processing
            futures = [
                _process_obj_chunk_indices.options(
                    scheduling_strategy=PlacementGroupSchedulingStrategy(
                        placement_group=pg,
                        placement_group_bundle_index=i,
                    )
                ).remote(
                    c,
                    expr_ref,
                    dict_ref,
                    len(self.constrs_gps_r),
                    len(self.constrs_gps_d),
                )
                for i, c in enumerate(chunks)
            ]

            # block on results
            results = ray.get(futures)

        # reconstruct groups
        obj_r: list[cp.Expression] = []
        for g_idx in range(len(self.constrs_gps_r)):
            # Gather all expression objects using the returned indices
            group_terms = [expr_list[i] for res in results for i in res[0][g_idx]]
            obj_r.append(
                t.cast(cp.Expression, cp.sum(group_terms) if group_terms else cp.Constant(0))
            )

        obj_d: list[cp.Expression] = []
        for g_idx in range(len(self.constrs_gps_d)):
            group_terms = [expr_list[i] for res in results for i in res[1][g_idx]]
            obj_d.append(
                t.cast(cp.Expression, cp.sum(group_terms) if group_terms else cp.Constant(0))
            )

        self._obj_expr_r, self._obj_expr_d = obj_r, obj_d

        return self._obj_expr_r, self._obj_expr_d


@ray.remote
def _process_obj_chunk_indices(
    indices: NDArray[np.int64],
    expr_list_ref: list[cp.Expression],
    var_id_pos_to_idx: dict[VarInfoT, list[tuple[int, int]]],
    num_r: int,
    num_d: int,
) -> tuple[list[list[int]], list[list[int]]]:
    """
    Returns only integer indices of the expressions.
    """
    # Local accumulation of INDICES
    local_r_idx: list[list[int]] = [[] for _ in range(num_r)]
    local_d_idx: list[list[int]] = [[] for _ in range(num_d)]

    for idx in indices:
        # Access the object from the shared reference
        obj = expr_list_ref[idx]

        var_id_pos_list = get_var_id_pos_list_from_tree(obj)

        if not var_id_pos_list:
            if num_r > 0:
                local_r_idx[0].append(idx)
            elif num_d > 0:
                local_d_idx[0].append(idx)
            continue

        id_set = set(var_id_pos_to_idx[var_id_pos_list[0]])
        for var_id_pos in var_id_pos_list[1:]:
            id_set &= set(var_id_pos_to_idx[var_id_pos])

        if not id_set:
            raise ValueError(f"Objective term at index {idx} is not separable.")

        target: tuple[int, int] = list(id_set)[0]
        if target[0] == 0:
            local_r_idx[target[1]].append(idx)
        else:
            local_d_idx[target[1]].append(idx)

    return local_r_idx, local_d_idx
