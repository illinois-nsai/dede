#!/usr/bin/env python3

import os
import pickle
import sys
import traceback

from benchmark_helpers import DEDE_FORM_HYPERPARAMS, get_args_and_problems, print_
from lib.algorithms import DeDeFormulation, Objective
from lib.problem import Problem

TOP_DIR = "dede-form-logs"
HEADERS = [
    "problem",
    "num_nodes",
    "num_edges",
    "traffic_seed",
    "scale_factor",
    "tm_model",
    "num_commodities",
    "total_demand",
    "algo",
    "num_paths",
    "edge_disjoint",
    "dist_metric",
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


def benchmark(problems, output_csv, args):
    num_paths, edge_disjoint, dist_metric = DEDE_FORM_HYPERPARAMS

    # build subproblems
    problem_name, topo_fname, tm_fname = problems[0]
    problem = Problem.from_file(topo_fname, tm_fname)
    df = DeDeFormulation(
        objective=Objective.get_obj_from_str(args.obj),
        num_paths=num_paths,
        edge_disjoint=edge_disjoint,
        dist_metric=dist_metric,
    )

    # warm-up
    for problem_name, topo_fname, tm_fname in problems[: args.warmup]:
        problem = Problem.from_file(topo_fname, tm_fname)
        df.solve(
            problem,
            num_cpus=args.num_cpus,
            rho=args.rho,
            num_iter=args.warmup_admm_steps,
            num_fix_iter=args.fix_steps,
        )

    # after warm-up
    with open(output_csv, "a") as results:
        print_(",".join(HEADERS), file=results)
        for problem_name, topo_fname, tm_fname in problems[args.warmup :]:
            problem = Problem.from_file(topo_fname, tm_fname)
            print_(problem_name, tm_fname)
            traffic_seed = problem.traffic_matrix.seed
            total_demand = problem.total_demand
            print_("traffic seed: {}".format(traffic_seed))
            print_("traffic scale factor: {}".format(problem.traffic_matrix.scale_factor))
            print_("traffic matrix model: {}".format(problem.traffic_matrix.model))
            print_("total demand: {}".format(total_demand))

            run_dir = os.path.join(
                TOP_DIR,
                problem_name,
                "{}-{}".format(traffic_seed, problem.traffic_matrix.model),
            )
            if not os.path.exists(run_dir):
                os.makedirs(run_dir)

            try:
                print_(
                    "\nPath formulation, objective {}, {} paths, edge disjoint {}, dist metric {}".format(
                        args.obj, num_paths, edge_disjoint, dist_metric
                    )
                )

                problem = Problem.from_file(topo_fname, tm_fname)
                df.solve(
                    problem,
                    num_cpus=args.num_cpus,
                    rho=args.rho,
                    num_iter=args.admm_steps,
                    num_fix_iter=args.fix_steps,
                )

                with open(
                    os.path.join(
                        run_dir,
                        "{}-dede-formulation_objective-{}_{}-paths_edge-disjoint-{}_dist-metric-{}_sol-mat.pkl".format(
                            problem_name, args.obj, num_paths, edge_disjoint, dist_metric
                        ),
                    ),
                    "wb",
                ) as w:
                    pickle.dump(df.sol_mat, w)

                result_line = PLACEHOLDER.format(
                    problem_name,
                    len(problem.G.nodes),
                    len(problem.G.edges),
                    traffic_seed,
                    problem.traffic_matrix.scale_factor,
                    problem.traffic_matrix.model,
                    len(problem.commodity_list),
                    total_demand,
                    "path_formulation",
                    num_paths,
                    edge_disjoint,
                    dist_metric,
                    args.obj,
                    df.get_fix_obj(),
                    df.runtime,
                    args.num_cpus,
                    args.rho,
                    args.admm_steps,
                    args.fix_steps,
                )
                print_(result_line, file=results)

            except Exception:
                print_(
                    "Path formulation, objective {}, {} paths, edge disjoint {}, dist metric {}, Problem {}, traffic seed {}, traffic model {} failed".format(
                        args.obj,
                        num_paths,
                        edge_disjoint,
                        dist_metric,
                        problem_name,
                        traffic_seed,
                        problem.traffic_matrix.model,
                    )
                )
                traceback.print_exc(file=sys.stdout)


if __name__ == "__main__":
    if not os.path.exists(TOP_DIR):
        os.makedirs(TOP_DIR)

    args, output_csv, problems = get_args_and_problems(OUTPUT_CSV_TEMPLATE)

    if args.dry_run:
        print("Problems to run:")
        for problem in problems:
            print(problem)
    else:
        benchmark(problems, output_csv, args)
