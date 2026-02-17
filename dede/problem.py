import os
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

from .constraints_utils import breakdown_constr
from .subproblems_wrap import SubproblemsWrap
from .utils import (
    VarInfoT,
    expand_expr,
    get_var_id_pos_list_from_cone,
    get_var_id_pos_list_from_linear,
)

KeyT = tuple[float, int]
ObjectiveT = t.Union[cp.Maximize, cp.Minimize]
ConstraintT = t.Union[Equality, Zero, Inequality]


class SubprobCache:
    """Cache subproblems."""

    def __init__(self):
        self.rho: t.Optional[float] = None
        self.num_cpus: t.Optional[int] = None
        self.probs: t.Optional[list[ray.actor.ActorProxy[SubproblemsWrap]]] = None
        self.param_idx_r: list[list[int]] = []
        self.param_idx_d: list[list[int]] = []

    def invalidate(self):
        self.rho = None
        self.num_cpus = None
        self.probs = None
        self.param_idx_r, self.param_idx_d = [], []

    @property
    def key(self) -> t.Optional[KeyT]:
        if self.rho is None or self.num_cpus is None:
            return None
        return (self.rho, self.num_cpus)

    @classmethod
    def make_key(cls, rho: float, num_cpus: int) -> KeyT:
        return (rho, num_cpus)


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
        start = time.time()

        # breakdown constraints
        constrs_r_converted = [self._convert_inequality(constr) for constr in resource_constraints]
        constrs_d_converted = [self._convert_inequality(constr) for constr in demand_constraints]

        self._constrs_r = breakdown_constr(constrs_r_converted, 0)
        self._constrs_d = breakdown_constr(constrs_d_converted, 1)

        # init subprob cache
        self._subprob_cache = SubprobCache()

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

        # get objective groups
        self._obj_expr_r, self._obj_expr_d = self.group_objective()
        end = time.time()
        print("init time:", end - start)

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
        enable_dede: bool = True,
        num_cpus: t.Optional[int] = None,
        rho: t.Optional[float] = None,
        num_iter: t.Optional[int] = None,
        *args,
        **kwargs,
    ) -> np.floating[t.Any]:
        """Compiles and solves the original problem.
        Args:
            enable_dede: whether to decouple and decompose with DeDe
            num_cpus: number of CPUs to use; all the CPUs available if None
            rho: rho value in ADMM; 1 if None
            num_iter: ADMM iterations; stop under < 1% improvement if None
        """
        # solve the original problem
        if not enable_dede:
            start = time.time()
            super(Problem, self).solve(*args, **kwargs)
            end = time.time()
            self._total_time = end - start

            coeff = 1 if self._problem_type == Minimize else -1
            return t.cast(np.floating[t.Any], coeff * self.value)

        # initialize num_cpus, rho
        if num_cpus is None:
            if self._subprob_cache.num_cpus is None:
                num_cpus = os.cpu_count() or 1
            else:
                num_cpus = self._subprob_cache.num_cpus
        if rho is None:
            if self._subprob_cache.rho is None:
                rho = 1
            else:
                rho = self._subprob_cache.rho
        # check whether num_cpus is more than all available
        if num_cpus > (os.cpu_count() or 1):
            raise ValueError(f"{num_cpus} CPUs exceeds upper limit of {os.cpu_count()}.")

        # check whether settings has been changed
        key = self._subprob_cache.make_key(rho, num_cpus)
        if key != self._subprob_cache.key:
            # invalidate old settings
            self._subprob_cache.invalidate()
            self._subprob_cache.rho = rho
            # initialize ray
            ray.shutdown()
            self._subprob_cache.num_cpus = num_cpus
            ray.init(num_cpus=num_cpus)
            # store subproblem in last solution
            self._subprob_cache.probs = self.get_subproblems(num_cpus, rho)
            # store parameter index in z solutions for x problems
            self._subprob_cache.param_idx_r, self._subprob_cache.param_idx_d = self.get_param_idx()
            # get demand solution
            self.sol_d = np.hstack(
                ray.get([prob.get_solution_d.remote() for prob in self._subprob_cache.probs])
            )

        assert self._subprob_cache.probs is not None

        # update parameter values
        param_id_to_value = {
            param_id: t.cast(NDArray[np.floating[t.Any]], param.value)
            for param_id, param in self.param_id_to_param.items()
        }
        ray.get(
            [prob.update_parameters.remote(param_id_to_value) for prob in self._subprob_cache.probs]
        )

        # solve problem
        # use num_iter if specifed
        # otherwise, stop under < 1% improvement or reach 10000 upper limit
        i, aug_lgr, aug_lgr_old = 0, 1, 2
        while (num_iter is not None and i < num_iter) or (
            num_iter is None
            and i < (num_iter or 10000)
            and (i < 2 or abs((aug_lgr - aug_lgr_old) / aug_lgr_old) > 0.01)
        ):
            # initialize start time, iteration, augmented Lagrangian
            start, i, aug_lgr_old, aug_lgr = time.time(), i + 1, aug_lgr, 0

            # resource allocation
            aug_lgr += sum(
                ray.get(
                    [
                        prob.solve_r.remote(self.sol_d[param_idx], *args, **kwargs)
                        for prob, param_idx in zip(
                            self._subprob_cache.probs, self._subprob_cache.param_idx_r
                        )
                    ]
                )
            )
            self.sol_r: NDArray[np.floating[t.Any]] = np.hstack(
                ray.get([prob.get_solution_r.remote() for prob in self._subprob_cache.probs])
            )

            # demand allocation
            aug_lgr += sum(
                ray.get(
                    [
                        prob.solve_d.remote(self.sol_r[param_idx], *args, **kwargs)
                        for prob, param_idx in zip(
                            self._subprob_cache.probs, self._subprob_cache.param_idx_d
                        )
                    ]
                )
            )
            self.sol_d: NDArray[np.floating[t.Any]] = np.hstack(
                ray.get([prob.get_solution_d.remote() for prob in self._subprob_cache.probs])
            )

            print("iter%d: end2end time %.4f, aug_lgr=%.4f" % (i, time.time() - start, aug_lgr))

        self.populate_vars_with_solution()
        coeff = 1 if self._problem_type == Minimize else -1
        return t.cast(
            np.floating[t.Any],
            coeff * sum(ray.get([prob.get_obj.remote() for prob in self._subprob_cache.probs])),
        )

    def populate_vars_with_solution(self) -> None:
        """Fills problem variables with computed solutions."""
        var_id_to_var = {var.id: var for var in self.variables()}
        for var in self.variables():
            var.value = np.zeros(var.shape)

        assert self._subprob_cache.probs is not None

        local_sol_idx: list[list[VarInfoT]] = ray.get(
            [prob.get_local_solution_idx.remote() for prob in self._subprob_cache.probs]
        )
        local_sol: list[NDArray[np.floating[t.Any]]] = ray.get(
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
    def _get_constr_dict(cls, constrs: list[cp.Constraint]) -> dict[cp.Constraint, list[VarInfoT]]:
        """Get a mapping of constraint to its var_id_pos_list."""
        constr_to_var_id_pos_list: dict[cp.Constraint, list[VarInfoT]] = {}
        for constr in constrs:
            # [constr] = get_var_id_pos_list_from_linear(constr.expr, self._solver)
            constr_to_var_id_pos_list[constr] = get_var_id_pos_list_from_linear(constr.expr)
        return constr_to_var_id_pos_list

    @classmethod
    def _group_constrs(
        cls, constrs: list[cp.Constraint], constr_dict: dict[cp.Constraint, list[VarInfoT]]
    ) -> list[list[cp.Constraint]]:
        """Group constraints into non-overlapped groups with union-find."""
        parents: list[int] = np.arange(len(constrs)).tolist()

        def find(x: int) -> int:
            if x == parents[x]:
                return x
            parents[x] = find(parents[x])
            return parents[x]

        def union(x1: int, x2: int) -> None:
            parent_x1 = find(x1)
            parent_x2 = find(x2)
            if parent_x1 != parent_x2:
                parents[parent_x2] = parent_x1

        var_id_pos_to_i: dict[VarInfoT, int] = {}
        for i, constr in enumerate(constrs):
            for var_id_pos in constr_dict[constr]:
                if var_id_pos in var_id_pos_to_i:
                    union(var_id_pos_to_i[var_id_pos], i)
                var_id_pos_to_i[var_id_pos] = i

        parent_to_constrs: dict[int, list[cp.Constraint]] = defaultdict(list)
        for i, parent in enumerate(parents):
            parent_to_constrs[find(parent)].append(constrs[i])
        return list(parent_to_constrs.values())

    def get_subproblems(
        self, num_cpus: int, rho: float
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

        # build actors with subproblems
        probs: list[ray.actor.ActorProxy[SubproblemsWrap]] = []
        for cpu in range(num_cpus):
            # get constraint idx for the group
            idx_r: list[int] = constrs_gps_idx_r[cpu::num_cpus].tolist()
            idx_d: list[int] = constrs_gps_idx_d[cpu::num_cpus].tolist()
            # get constraints group
            constrs_r = [self.constrs_gps_r[j] for j in idx_r]
            constrs_d = [self.constrs_gps_d[j] for j in idx_d]
            # get obj groups
            obj_r = [self._obj_expr_r[j] for j in idx_r]
            obj_d = [self._obj_expr_d[j] for j in idx_d]
            # get var_id_to_pos_list
            var_id_to_pos_r = [
                [self.constr_dict_r[constr] for constr in constrs] for constrs in constrs_r
            ]
            var_id_to_pos_d = [
                [self.constr_dict_d[constr] for constr in constrs] for constrs in constrs_d
            ]
            # build subproblems
            actor = ray.remote(SubproblemsWrap)
            probs.append(
                actor.remote(
                    idx_r,
                    idx_d,
                    obj_r,
                    obj_d,
                    constrs_r,
                    constrs_d,
                    var_id_to_pos_r,
                    var_id_to_pos_d,
                    var_id_pos_set_r,
                    var_id_pos_set_d,
                    rho,
                )
            )
        return probs

    def get_param_idx(self) -> tuple[list[list[int]], list[list[int]]]:
        """Get parameter z index in last solution."""
        assert self._subprob_cache.probs is not None

        # map var_id_pos in the big resource solution list
        sol_idx_r: list[list[VarInfoT]] = ray.get(
            [prob.get_solution_idx_r.remote() for prob in self._subprob_cache.probs]
        )

        sol_idx_dict_r: dict[VarInfoT, int] = {}
        idx = 0
        for sol_idx in sol_idx_r:
            for var_id_pos in sol_idx:
                sol_idx_dict_r[var_id_pos] = idx
                idx += 1

        # map var_id_pos in the big demand solution list
        sol_idx_d: list[list[VarInfoT]] = ray.get(
            [prob.get_solution_idx_d.remote() for prob in self._subprob_cache.probs]
        )
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

    def group_objective(self) -> tuple[list[cp.Expression], list[cp.Expression]]:
        """Split objective into corresponding constraint groups"""
        var_id_pos_to_idx: dict[VarInfoT, list[tuple[int, int]]] = defaultdict(list)
        for i, constrs_gps, constr_dict in zip(
            [0, 1],
            [self.constrs_gps_r, self.constrs_gps_d],
            [self.constr_dict_r, self.constr_dict_d],
        ):
            for j, constrs in enumerate(constrs_gps):
                for constr in constrs:
                    for var_id_pos in constr_dict[constr]:
                        var_id_pos_to_idx[var_id_pos].append((i, j))

        obj_r: list[cp.Expression] = [cp.Constant(0) for _ in self.constrs_gps_r]
        obj_d: list[cp.Expression] = [cp.Constant(0) for _ in self.constrs_gps_d]
        for obj in expand_expr(self.objective.expr):
            var_id_pos_list = get_var_id_pos_list_from_cone(obj, self._solver)
            if not var_id_pos_list:
                if len(obj_r) > 0:
                    obj_r[0] += obj
                elif len(obj_d) > 0:
                    obj_d[0] += obj
                continue

            id_set = set(var_id_pos_to_idx[var_id_pos_list[0]])
            for var_id_pos in var_id_pos_list[1:]:
                id_set = id_set & set(var_id_pos_to_idx[var_id_pos])
            if not id_set:
                raise ValueError("Objective not separable.")

            idx = list(id_set)[0]
            if idx[0] == 0:
                obj_r[idx[1]] += obj
            else:
                obj_d[idx[1]] += obj

        return obj_r, obj_d
