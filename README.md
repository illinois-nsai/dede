![dede logo](assets/dede_logo.svg)
======
![PyPI - version](https://img.shields.io/pypi/v/dede?label=PyPI%20package)
![PyPI - downloads](https://img.shields.io/pypi/dm/dede?label=PyPI%20downloads)

DeDe is a general, scalable, and theoretically grounded optimization framework that accelerates large-scale resource allocation problems through a *decouple-and-decompose* approach.

## Getting started

### Hardware requirements
- Linux OS (e.g., Ubuntu 24.04)
- A multi-core CPU instance

### Dependencies
- Python >= 3.8
- `g++` (required by `cvxpy`)
- (optional) install `pytest` with `pip install -U pytest`

## Installation
We have made DeDe available as a PyPI package! You can simply install it using pip:
```
pip install dede
```
- We recommend creating a Python virtual environment (e.g., venv or Conda) before installation.

## Modifications
If you want to modify DeDe's source code and use that as your package instead, navigate
to the project root directory and run:
```
pip install -e .
```
- After making modifications, there is no need to rerun this command, as the package was installed in editable (`-e`) mode.
- Once again, we recommend creating a Python virtual environment beforehand.

## Code structure

```shell
.
├── dede/                       # core source code
├── tests/                      # test suite
│     └── test_dede.py
└── examples/                   # example use cases
      ├── traffic_engineering/
      ├── cluster_scheduling/
      └── load_balancing/
```

## Using DeDe
DeDe adopts a familiar interface from `cvxpy`, e.g., `Variable(.)`, `Minimize(.)`.

Key differences in DeDe:
- DeDe requires specifying separate `resource_constraints` and `demand_constraints` when constructing a problem.
- The `solve(.)` method includes additional parameters:
  - `enable_dede`: enables DeDe if `True`; defaults to `cvxpy` if `False`.
  - `num_cpus`: number of CPU cores (defaults to all available cores).
  - `rho`: ADMM parameter acting like a ``learning rate''.
  - `num_iter`: maximum number of iterations; if not specified, DeDe stops if the accuracy improvement falls below 1%.

> ⚠️ **Work in progress:**
>
> This high-level interface is designed for ease of use, but performance enhancements are still underway. For optimal performance, we currently recommend lower-level APIs, as demonstrated in the [example use cases](examples/README.md).

### Toy examples
A toy example for resource allocation with DeDe is as follows:
```python
import dede as dd
N, M = 100, 100

# Create allocation variables
x = dd.Variable((N, M), nonneg=True)

# Create the constraints
resource_constraints = [x[i,:].sum() >= i for i in range(N)]
demand_constraints = [x[:,j].sum() <= j for j in range(M)]

# Create an objective
objective = dd.Minimize(x.sum())

# Construct the problem
prob = dd.Problem(objective, resource_constraints, demand_constraints)

# Solve the problem with DeDe on 4 CPU cores
print(prob.solve(num_cpus=4, solver=dd.ECOS))
```

Another toy example is provided in `tests/test_dede.py`. To test these examples quickly, from the project root directory, run
```
./tests/test_dede.py
```
or, if `pytest` is installed:
```
pytest
```
Example output screenshots are provided in the `assets` folder.

## Example use cases of DeDe
We provide three example applications of DeDe in the research paper:
- **Traffic engineering**: a network flow optimization problem.
- **Cluster scheduling**: a resource allocation problem in cluster computing.
- **Load balancing**: a query balancing problem in distributed stores.

These examples are implemented using the lower-level APIs of DeDe, which
currently offer greater efficiency than the high-level APIs.
Please see [examples/README.md](examples/README.md) for more details.

## Citation
If you use DeDe in your research, please cite our paper:
```
@inproceedings{dede,
    title={{Decouple and Decompose: Scaling Resource Allocation with DeDe}},
    author={Xu, Zhiying and Yu, Minlan and Yan, Francis Y.},
    booktitle={Proceedings of the USENIX OSDI 2025 Conference},
    month=jul,
    year={2025}
}
```
