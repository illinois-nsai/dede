import argparse
import os
import sys


# the same randomNums as POP
randomNums = [
    0.7275636800328681, 0.6832234717598454, 0.30871945533265976, 0.27707849007413665, 0.6655489517945736,
    0.9033722646721782, 0.36878291341130565, 0.2757480694417024, 0.46365357580915334, 0.7829017787900358,
    0.9193277828687169, 0.43649097442328655, 0.7499061812554475, 0.38656687435934867, 0.17737847790937833,
    0.5943499108896841, 0.20976756886633208, 0.825965871887821, 0.17221793768785243, 0.5874273817862956,
    0.7512804067674601, 0.5710403484148672, 0.5800248845020607, 0.752509948590651, 0.03141823882658079,
    0.35791991947712865, 0.8177969308356393, 0.41768754675291875, 0.9740356814958814, 0.7134062578232291,
    0.48057451655643435, 0.2916564974118041, 0.9498601346594666, 0.8204918233863466, 0.636644547856282,
    0.3691214939418974, 0.3602548753661351, 0.4346610851406103, 0.4573170944447694, 0.4726884208758554,
    0.4690225206155686, 0.8273224240149951, 0.15103155452875827, 0.8338662354441657, 0.46030637266116115,
    0.2805719916728102, 0.195964207423156, 0.17927344087491737, 0.8656867963273522, 0.48659066182619404,
    0.4209717066310287, 0.632869791353279, 0.6998586450671721, 0.31532845005767296, 0.5716203055299767,
    0.3710122896837609, 0.8718145959648387, 0.805730942661998, 0.6162136850351787, 0.3743593560890043,
    0.6972487292697295, 0.908614580207571, 0.19614707188185154, 0.8091248167277394, 0.6279332754134727,
    0.4633992710283451, 0.30557915566744887, 0.5399094342593693, 0.6351110144563881, 0.12625782329876534,
    0.49732689247592055, 0.027314166285965835, 0.03648451669024966, 0.48384385495430515, 0.655053381109821,
    0.39576628017124216, 0.940172465685381, 0.3846108439172914, 0.6462319787976428, 0.770465637773941,
    0.8940427958184088, 0.5988370371450177, 0.9760344716184084, 0.077085112935252, 0.7751206959271756,
    0.2788223024987677, 0.8992053297295577, 0.3738361436205424, 0.4313095551354611, 0.33232427838474854,
    0.3151474713673178, 0.44273594208622036, 0.096450915880824, 0.7451533062153856, 0.17085973788289754,
    0.8053907199213823, 0.13978959528686086, 0.09294681694145557, 0.02702986688213338, 0.5483346917317515,
]


def check_gurobi_license():
    if not os.system('gurobi_cl --license'):
        return True
    else:
        return False


def get_args():
    if not check_gurobi_license():
        raise Exception("Gurobi license not found")

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-servers",
        type=int,
        default=32,
        help="number of shard servers",
    )
    parser.add_argument(
        "--num-shards",
        type=int,
        default=256,
        help="number of shards",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.1,
        help="epsilon in load balancing",
    )
    parser.add_argument(
        "--memory-limit",
        type=int,
        default=16,
        help="upper limit for number of shards per server",
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
        default=10,
        help="number of admm steps",
    )
    parser.add_argument(
        "--rho",
        type=float,
        default=10,
        help="rho in ADMM",
    )
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=os.cpu_count(),
        help="number of CPU cores",
    )
    parser.add_argument(
        "--search-limit",
        type=int,
        default=48,
        help='search limit among top shards',
    )

    args = parser.parse_args()
    return args


def print_(*args, file=None):
    if file is None:
        file = sys.stdout
    print(*args, file=file)
    file.flush()
