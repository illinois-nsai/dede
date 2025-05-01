#!/usr/bin/env python3

from benchmark_helpers import get_args, print_

import os
import pickle
import json
import cvxpy as cp

from lib.scheduler import Scheduler
from lib.utils import get_policy


TOP_DIR = "dede-form-logs"
HEADERS = [
    "num_workers",
    "num_jobs",
    "objective",
    "obj_val",
    "runtime",
    "num_cpu",
    "rho",
    "admm_steps",
    "fix_steps",
]
PLACEHOLDER = ",".join("{}" for _ in HEADERS)

OUTPUT_CSV_TEMPLATE = "dede-form-{}.csv"


def benchmark(args, output_csv):
    policy = get_policy(args.obj)
    cluster_spec = json.load(open(args.cluster_spec_file, 'r'))
    if args.num_worker_types > 0:
        cluster_spec = {
            worker_type: n for worker_type,
            n in cluster_spec.items() if int(
                worker_type.split('_')[1]) < args.num_worker_types}
    sched = Scheduler(
        policy,
        throughputs_file=args.throughputs_file,
        enable_dede=True, num_cpus=args.num_cpus, rho=args.rho,
        warmup=args.warmup, warmup_admm_steps=args.warmup_admm_steps, admm_steps=args.admm_steps,
        fix_steps=args.fix_steps,
    )

    run_dir = os.path.join(
        TOP_DIR,
        args.obj,
    )
    if not os.path.exists(run_dir):
        os.makedirs(run_dir)

    with open(output_csv, "a") as results:
        print_(",".join(HEADERS), file=results)
        sched.simulate(
            cluster_spec,
            lam=args.lam,
            generate_multi_gpu_jobs=True,
            generate_multi_priority_jobs=True,
            simulate_steady_state=True,
            num_total_jobs=1e5,
            max_iter=args.max_iter,
            results_file=results,
            results_folder=run_dir,
        )


if __name__ == '__main__':
    if not os.path.exists(TOP_DIR):
        os.makedirs(TOP_DIR)

    args, output_csv = get_args(OUTPUT_CSV_TEMPLATE)
    benchmark(args, output_csv)
