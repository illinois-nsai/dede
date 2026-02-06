import argparse
import os
import sys

from lib.policies.objective import OBJ_STRS


def check_gurobi_license():
    if not os.system("gurobi_cl --license"):
        return True
    else:
        return False


def get_args(formatted_fname_template):
    if not check_gurobi_license():
        raise Exception("Gurobi license not found")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--obj",
        type=str,
        choices=OBJ_STRS,
        required=True,
        help="objective function",
    )
    parser.add_argument(
        "--cluster-spec-file",
        type=str,
        default="data/cluster_spec.json",
        help="number of workers for each worker type",
    )
    parser.add_argument(
        "--throughputs-file",
        type=str,
        default="data/simulation_throughputs.npy",
        help="throughput file",
    )
    parser.add_argument(
        "--lam",
        type=int,
        default=3600,
        help="lambda in poisson process for jobs",
    )
    parser.add_argument(
        "--num-worker-types",
        type=int,
        default=10,
        help="number of worker types",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=100,
        help="max iterations in scheduling simulation",
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
        default=100,
        help="number of admm steps during warmup",
    )
    parser.add_argument(
        "--admm-steps",
        type=int,
        default=20,
        help="number of admm steps",
    )
    parser.add_argument(
        "--fix-steps",
        type=int,
        default=5,
        help="number of steps to fix violation",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=0.1,
        help="rho in ADMM",
    )
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=os.cpu_count(),
        help="number of CPU cores",
    )

    args = parser.parse_args()
    formatted_fname_substr = formatted_fname_template.format(args.obj)
    return args, formatted_fname_substr


def print_(*args, file=None):
    if file is None:
        file = sys.stdout
    print(*args, file=file)
    file.flush()
