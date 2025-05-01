# DeDe

DeDe is a general, scalable, and theoretically grounded framework that accelerates resource allocation through **decouple** and **decompose**.

## Getting started

### Hardware requirements
- Linux OS
- A multi-core CPU instance

### Dependencies
- Python >= 3.8
- Run `pip install dede` to install `dede`

## Code structure

```
.
├── dede                         # source code for dede
├── tests                        # test code for dede
│     └── test_dede.py
└── examples                     # examples for dede (details in examples/README.md)
      ├── traffic_engineering
      ├── cluster_scheduling
      └── load_balancing
```

## Running DeDe
DeDe borrows the interface from `cvxpy` and inherits most of its methods, e.g. `Variable(.)`, `Minimize(.)`. 

Different from `cvxpy`,

- DeDe requires separate `resource_constraints` and `demand_constraints` when constructing a problem.
- DeDe has additional arguments in the `solve(.)` methods:
  - `enable_dede`: use DeDe if True; use `cvxpy` if False.
  - `num_cpus`: the number of CPU cores; if not specified, DeDe uses all available CPUs.
  - `rho`: the rho parameter in ADMM formulation.
  - `num_iter`: the number of iteration; if not specified, DeDe stops iterations when the improvement in accuracy is below 1%.

A toy example for resource allocation with DeDe is as follows. To run this example, run `PYTHONPATH=<PROJECT_ROOT> pytest` directly.
```
import dede as dd
N, M = 100, 100

# Create allocation variables.
x = dd.Variable((N, M), nonneg=True)

# Create the constraints.
resource_constraints = [x[i,:].sum() >= i for i in range(N)]
demand_constraints = [x[:,j].sum() <= j for j in range(M)]

# Create an objective.
objective = dd.Minimize(x.sum())

# Construct the problem.
prob = dd.Problem(objective, resource_constraints, demand_constraints)

# Solve the problem with DeDe on a 4-core CPU.
print(prob.solve(num_cpus=4, solver=dd.ECOS))

# Solve the problem with cvxpy
print(prob.solve(enable_dede=False))
```

## Running examples of DeDe
Please refer to [README.md](examples/README.md) for details.

## Citation
If you use our code in your research, please cite our paper:
```
@inproceedings{dede,
    title={Decouple and Decompose: Scaling Resource Allocation through a Different Lens},
    author={Xu, Zhiying and Yu, Minlan and Yan, Francis Y.},
    booktitle={Proceedings of the USENIX OSDI 2025 Conference},
    month=jul,
    year={2025}
}
```
