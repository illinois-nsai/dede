#!/usr/bin/env python3

from benchmark_helpers import get_args, print_, randomNums

import os
import sys
import math
import numpy as np
import pickle

from lib.dede_formulation import DeDeFormulation

TOP_DIR = "dede-form-logs"
HEADERS = [
    "num_servers",
    "num_shards",
    "epsilon",
    "memory_limit",
    "search_limit",
    "random_number",
    "obj_val",
    "runtime",
    "num_cpu",
    "rho",
    "admm_steps",
]
PLACEHOLDER = ",".join("{}" for _ in HEADERS)


def get_shardLoads(num_shards, randomNum):
    shardLoads = np.zeros(num_shards)
    zipfValue = 0.25 + randomNum * 0.5
    for shardNum in range(num_shards):
        load = int(round(1000000.0 * (1.0 / math.pow(shardNum / num_shards * 1024 + 1, zipfValue)))) / 10000.0
        shardLoads[shardNum] = load
    return shardLoads


def benchmark(args):
    df = DeDeFormulation(args.num_servers, args.num_shards, args.epsilon, args.memory_limit, args.search_limit)
    # warm-up
    for i, randomNum in enumerate(randomNums[:args.warmup]):
        shardLoads = get_shardLoads(args.num_shards, randomNum)
        df.solve(shardLoads, num_cpus=args.num_cpus, rho=args.rho, num_iter=args.warmup_admm_steps)

    result_list = []
    with open("dede-form.csv", "a") as results:
        print_(",".join(HEADERS), file=results)
        for i, randomNum in enumerate(randomNums[args.warmup:]):
            shardLoads = get_shardLoads(args.num_shards, randomNum)
            print(f'Iteration {i + args.warmup}, total shards {shardLoads.sum()}')
            obj = df.solve(shardLoads, num_cpus=args.num_cpus, rho=args.rho, num_iter=args.admm_steps)

            with open(
                os.path.join(
                    TOP_DIR,
                    "{}_sol-mat.pkl".format(i + args.warmup
                                            ),
                ),
                "wb",
            ) as w:
                pickle.dump(df.sol_mat, w)

            result_line = PLACEHOLDER.format(
                args.num_servers,
                args.num_shards,
                args.epsilon,
                args.memory_limit,
                args.search_limit,
                randomNum,
                obj,
                df.runtime,
                args.num_cpus,
                args.rho,
                args.admm_steps,
            )
            print_(result_line, file=results)
            result_list.append([obj, df.runtime])
    result_list = np.array(result_list).mean(0)
    print(f'Average movement {result_list[0]:.2f}, average runtime  {result_list[1]:.2f}')


if __name__ == '__main__':
    if not os.path.exists(TOP_DIR):
        os.makedirs(TOP_DIR)

    args = get_args()
    benchmark(args)
