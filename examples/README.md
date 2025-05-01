# Example Use Cases of DeDe
This README provides instructions for running DeDe on three use cases — traffic engineering, cluster scheduling, and load balancing. We provide only toy and synthesized datasets here.

## Getting started

### Hardware requirements
- Linux OS
- A multi-core CPU instance

### Prerequisites
- Python >= 3.8
- `g++` (required by `cvxpy`)
- Run `conda env create -f environment.yml` and `conda activate dede-examples` to create and activate the Conda environment
    - [Miniconda](https://docs.anaconda.com/free/anaconda/install/index.html) or [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html) is required
- Acquire a Gurobi license from [Gurobi](https://www.gurobi.com/solutions/licensing/) and activate it with `grbgetkey [gurobi-license]`
    - Run `gurobi_cl` to verify the activation
- Python dependencies: `pip install networkx gurobipy`
- Install DeDe: `pip install dede`

## Running examples of DeDe

### Traffic engineering
```shell
$ cd traffic_engineering/
# maximize total flow
$ ./dede_form.py --obj total_flow --num-cpus 4 --rho 0.01
# minimize max link utilization
$ ./dede_form.py --obj min_max_link_util --num-cpus 4 --rho 1000
```
- Run `./dede_form.py --help` to see usage.
- Results will be saved in
   - `dede-form-[obj].csv`: performance numbers
   - `dede-form-logs`: directory with DeDe's solution matrices


### Cluster scheduling
```shell
$ cd cluster_scheduling/
# maximize min allocation
$ ./dede_form.py --obj max_min_fairness_perf --num-cpus 4 --rho 1
# maximize proportional fairness
$ ./dede_form.py --obj max_proportional_fairness --num-cpus 4 --rho 0.1
```

### Load balancing
```shell
$ cd load_balancing/
$ ./dede_form.py --num-cpus 4
```
