import argparse
import os
import sys
from collections import defaultdict
from glob import iglob

from lib.algorithms.abstract_formulation import OBJ_STRS
from lib.config import TM_DIR, TOPOLOGIES_DIR

PROBLEM_NAMES = [
    "B4.json",
    "UsCarrier.json",
    "Kdl.json",
    "ASN2k.json",
]
TM_MODELS = [
    "real",
]
SCALE_FACTORS = [1.0]

DEDE_FORM_HYPERPARAMS = (4, True, "min-hop")
PATH_FORM_HYPERPARAMS = (4, True, "min-hop")

PROBLEM_NAMES_AND_TM_MODELS = [
    (prob_name, tm_model) for prob_name in PROBLEM_NAMES for tm_model in TM_MODELS
]

PROBLEMS = []
GROUPED_BY_PROBLEMS = defaultdict(list)
HOLDOUT_PROBLEMS = []
GROUPED_BY_HOLDOUT_PROBLEMS = defaultdict(list)

for problem_name in PROBLEM_NAMES:
    if problem_name.endswith(".graphml"):
        topo_fname = os.path.join(TOPOLOGIES_DIR, "topology-zoo", problem_name)
    else:
        topo_fname = os.path.join(TOPOLOGIES_DIR, problem_name)
    for model in TM_MODELS:
        for tm_fname in iglob("{}/{}/{}*_traffic-matrix.pkl".format(TM_DIR, model, problem_name)):
            vals = os.path.basename(tm_fname)[:-4].split("_")
            _, traffic_seed, scale_factor = vals[1], int(vals[2]), float(vals[3])
            GROUPED_BY_PROBLEMS[(problem_name, model, scale_factor)].append((topo_fname, tm_fname))
            PROBLEMS.append((problem_name, topo_fname, tm_fname))
        for tm_fname in iglob(
            "{}/holdout/{}/{}*_traffic-matrix.pkl".format(TM_DIR, model, problem_name)
        ):
            vals = os.path.basename(tm_fname)[:-4].split("_")
            _, traffic_seed, scale_factor = vals[1], int(vals[2]), float(vals[3])
            GROUPED_BY_HOLDOUT_PROBLEMS[(problem_name, model, scale_factor)].append(
                (topo_fname, tm_fname)
            )
            HOLDOUT_PROBLEMS.append((problem_name, topo_fname, tm_fname))

GROUPED_BY_PROBLEMS = dict(GROUPED_BY_PROBLEMS)
for key, vals in GROUPED_BY_PROBLEMS.items():
    GROUPED_BY_PROBLEMS[key] = sorted(vals, key=lambda x: int(x[-1].split("_")[-3]))

GROUPED_BY_HOLDOUT_PROBLEMS = dict(GROUPED_BY_HOLDOUT_PROBLEMS)
for key, vals in GROUPED_BY_HOLDOUT_PROBLEMS.items():
    GROUPED_BY_HOLDOUT_PROBLEMS[key] = sorted(vals, key=lambda x: int(x[-1].split("_")[-3]))


def get_problems(args):
    problems = []
    for (
        (problem_name, tm_model, scale_factor),
        topo_and_tm_fnames,
    ) in GROUPED_BY_PROBLEMS.items():
        for topo_fname, tm_fname in topo_and_tm_fnames:
            if (
                ("all" in args.topos or problem_name in args.topos)
                and ("all" in args.tm_models or tm_model in args.tm_models)
                and ("all" in args.scale_factors or scale_factor in args.scale_factors)
            ):
                problems.append((problem_name, topo_fname, tm_fname))
    if not problems:
        raise Exception("Traffic matrices not found")
    return problems


def check_gurobi_license():
    if not os.system("gurobi_cl --license"):
        return True
    else:
        return False


def get_args_and_problems(
    formatted_fname_template,
    additional_args=[],
    **kwargs,
):
    if not check_gurobi_license():
        raise Exception("Gurobi license not found")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=False,
        help="list problems to run",
    )
    parser.add_argument(
        "--obj",
        type=str,
        choices=OBJ_STRS,
        required=True,
        help="objective function",
    )
    parser.add_argument(
        "--tm-models",
        type=str,
        choices=TM_MODELS + ["all"],
        nargs="+",
        default="all",
        help="traffic matrix model",
    )
    parser.add_argument(
        "--topos",
        type=str,
        choices=PROBLEM_NAMES + ["all"],
        nargs="+",
        default="B4.json",
        help="network topology",
    )
    parser.add_argument(
        "--scale-factors",
        type=lambda x: x if x == "all" else float(x),
        choices=SCALE_FACTORS + ["all"],
        nargs="+",
        default="all",
        help="traffic matrix scale factor",
    )

    # DeDe specific hyper-parameters
    parser.add_argument(
        "--warmup",
        type=int,
        default=1,
        help="number of warmup iterations",
    )
    parser.add_argument(
        "--warmup-admm-steps",
        type=int,
        default=1000,
        help="number of admm steps during warmup",
    )
    parser.add_argument(
        "--admm-steps",
        type=int,
        default=10,
        help="number of admm steps",
    )
    parser.add_argument(
        "--fix-steps",
        type=int,
        default=3,
        help="number of steps to fix violation",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=0.05,
        help="rho in ADMM",
    )
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=os.cpu_count(),
        help="number of CPU cores",
    )

    for add_arg in additional_args:
        name_or_flags, kwargs = add_arg[0], add_arg[1]
        parser.add_argument(name_or_flags, **kwargs)
    args = parser.parse_args()
    formatted_fname_substr = formatted_fname_template.format(args.obj)
    return args, formatted_fname_substr, get_problems(args)


def print_(*args, file=None):
    if file is None:
        file = sys.stdout
    print(*args, file=file)
    file.flush()
