from ..graph_utils import compute_in_or_out_flow
from collections import defaultdict
from enum import Enum, unique
import pickle
import re
import sys

EPS = 1e-5


OBJ_STRS = ["total_flow", "min_max_link_util"]


@unique
class Objective(Enum):
    TOTAL_FLOW = 0
    MIN_MAX_UTIL = 1
    MAX_CONCURRENT_FLOW = 2
    MIN_MAX_LINK_UTIL = 3
    COMPUTE_DEMAND_SCALE_FACTOR = 4
    TEAVAR = 5

    @classmethod
    def get_obj_from_str(cls, obj_str):
        if obj_str == "total_flow":
            return cls.TOTAL_FLOW
        elif obj_str == "mcf":
            return cls.MAX_CONCURRENT_FLOW
        elif obj_str == "min_max_link_util":
            return cls.MIN_MAX_LINK_UTIL
        elif obj_str == "compute_demand_scale_factor":
            return cls.COMPUTE_DEMAND_SCALE_FACTOR
        elif obj_str == "teavar":
            return cls.TEAVAR
        else:
            raise Exception("{} not supported".format(obj_str))


class AbstractFormulation(object):
    def __init__(self, objective, DEBUG=False, VERBOSE=False, out=None):
        if out is None:
            out = sys.stdout
        self._objective = objective
        self._warm_start_mode = False
        self.DEBUG = DEBUG
        self.VERBOSE = VERBOSE
        self.out = out

    def solve(self, problem, fixed_total_flows=[], **args):
        self._problem = problem
        self._solver = self._construct_lp(fixed_total_flows)
        return self._solver.solve_lp(**args)

    def solve_warm_start(self, problem):
        assert self._warm_start_mode
        for k, (_, _, d_k) in problem.sparse_commodity_list:
            constr = self._demand_constrs[k]
            constr.rhs = d_k
        self._solver.solve_lp()

    @property
    def problem(self):
        return self._problem

    @property
    def model(self):
        return self._solver.model

    @property
    def sol_dict(self):
        raise NotImplementedError(
            "sol_dict needs to be implemented in the subclass: {}".format(
                self.__class__
            )
        )

    @property
    def sol_mat(self):
        raise NotImplementedError(
            "sol_mat needs to be implemented in the subclass: {}".format(
                self.__class__)
        )

    ##########################
    # Private helper methods #
    ##########################
    def _print(self, *args):
        if self.DEBUG:
            print(*args, file=self.out)

    def _extract_inds_from_var_name(self, varName, var_group_name="f"):
        match = re.match(r"{}\[(\d+),(\d+)\]".format(var_group_name), varName)
        return int(match.group(1)), int(match.group(2))

    def _create_sol_dict(self, sol_dict_def, commodity_list):
        # Set zero-flow commodities to be empty lists
        sol_dict = {}
        sol_dict_no_def = dict(sol_dict_def)

        for commod_key in commodity_list:
            if commod_key in sol_dict_no_def:
                sol_dict[commod_key] = sol_dict_no_def[commod_key]
            else:
                sol_dict[commod_key] = []

        return sol_dict

    def _construct_lp(self, fixed_total_flows=[]):
        raise NotImplementedError(
            "_construct_lp needs to be implemented in the subclass: {}".format(
                self.__class__
            )
        )

    def _save_pkl(self, obj, fname):
        if fname.endswith(".pkl"):
            with open(fname, "wb") as w:
                pickle.dump(obj, w)

    def _save_txt(self, obj, fname):
        if fname.endswith(".txt"):
            with open(fname, "w") as w:
                print(obj, file=w)

    @property
    def runtime(self):
        raise NotImplementedError(
            "runtime needs to be implemented in the subclass: {}".format(
                self.__class__)
        )

    @property
    def obj_val(self):
        if not hasattr(self, "_obj_val"):
            if self._objective.value == Objective.TOTAL_FLOW.value:
                self._obj_val = self.total_flow
            elif self._objective.value == Objective.MAX_CONCURRENT_FLOW.value:
                self._obj_val = self.min_frac_flow
            elif (
                (self._objective.value == Objective.MIN_MAX_LINK_UTIL.value)
                or self._objective.value == Objective.COMPUTE_DEMAND_SCALE_FACTOR.value
            ):
                self._obj_val = self.max_link_util
            else:
                raise Exception(
                    "no support for other Objectives besides {}".format(
                        ", ".join(OBJ_STRS)
                    )
                )
        return self._obj_val

    @property
    def total_flow(self):
        if not hasattr(self, "_total_flow") or self.DEBUG:
            self._total_flow = 0.0
            for (_, (s_k, _, _)), flow_list in self.sol_dict.items():
                self._total_flow += compute_in_or_out_flow(flow_list, 0, {s_k})
        return self._total_flow

    @property
    def min_frac_flow(self):
        if not hasattr(self, "_min_frac_flow") or self.DEBUG:
            self.frac_flows = {}
            self._min_frac_flow = 1.0
            for commod_key, flow_list in self.sol_dict.items():
                _, (s_k, _, d_k) = commod_key
                if d_k < EPS:
                    continue
                out_flow = compute_in_or_out_flow(flow_list, 0, {s_k})
                self._min_frac_flow = min(self._min_frac_flow, out_flow / d_k)
                self.frac_flows[commod_key] = out_flow / d_k
        return self._min_frac_flow

    @property
    def max_link_util(self):
        if not hasattr(self, "_max_link_util") or self.DEBUG:
            self.link_vols = defaultdict(float)
            for _, flow_list in self.sol_dict.items():
                for edge, flow_vol in flow_list:
                    self.link_vols[edge] += flow_vol
            self.link_utils = {
                (u, v): flow_vol / self.problem.G[u][v]["capacity"]
                for (u, v), flow_vol in self.link_vols.items()
            }
            self._max_link_util = sorted(self.link_utils.values())[-1]

        return self._max_link_util
