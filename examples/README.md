# DeDe Examples
This README provides instructions for running DeDe on three case study examples. We provide only toy and synthesized datasets here.

## Getting started

### Hardware requirements
- Linux OS
- A multi-core CPU instance

### Dependencies
- Python >= 3.8
- Run `conda env create -f environment.yml` and `conda activate dede-examples` to create and activate the Conda environment
    - [Miniconda](https://docs.anaconda.com/free/anaconda/install/index.html) or [Anaconda](https://docs.anaconda.com/free/anaconda/install/index.html) is required
- Acquire a Gurobi license from [Gurobi](https://www.gurobi.com/solutions/licensing/) and activate it with `grbgetkey [gurobi-license]`
    - Run `gurobi_cl` to verify the activation
- Run `pip install dede` to install `dede`

## Running examples of DeDe

### Traffic engineering
```
$ cd traffic_engineering/
# maximize total flow
$ python dede_form.py --obj total_flow --num-cpus 4 --rho 0.01
# minimize max link utilization
$ python dede_form.py --obj min_max_link_util --num-cpus 4 --rho 1000
```
To show explanations on the input parameters, run `python dede_form.py --help`

Results will be saved in
- `dede-form-[obj].csv`: performance numbers
- `dede-form-logs`: directory with DeDe solution matrices


### Cluster scheduling
```
$ cd cluster_scheduling/
# maximize min allocation
$ python3 dede_form.py --obj max_min_fairness_perf --num-cpus 4 --rho 1
# maximize proportional fairness
$ python3 dede_form.py --obj max_proportional_fairness --num-cpus 4 --rho 0.1
```
To show explanations on the input parameters, run `python dede_form.py --help`

Results will be saved in
- `dede-form-[obj].csv`: performance numbers
- `dede-form-logs`: directory with DeDe solution matrices

### Load balancing
```
$ cd load_balancing/
$ python dede_form.py --num-cpus 4
```
To show explanations on the input parameters, run `python dede_form.py --help`

Results will be saved in
- `dede-form.csv`: performance numbers
- `dede-form-logs`: directory with DeDe solution matrices
