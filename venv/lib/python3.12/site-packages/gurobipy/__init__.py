"""
Python interface to the Gurobi Optimizer

The Gurobi Optimizer is a mathematical optimization software library for
solving mixed-integer linear, quadratic, and nonlinear optimization problems.

This package comes with a trial license that allows you to solve problems of
limited size. As a student or staff member of an academic institution, you
qualify for a free, full product license. For more information visit
https://www.gurobi.com/academia/academic-program-and-licenses/

For a commercial evaluation, you can request an evaluation license by
visiting https://www.gurobi.com/free-trial/.

Other useful resources to get started:

- Gurobi Documentation: https://docs.gurobi.com/
- Gurobi Community: https://support.gurobi.com/hc/en-us/community/topics/

A simple example, formulating and solving a mixed-integer linear program:

#  maximize
#        x +   y + 2 z
#  subject to
#        x + 2 y + 3 z <= 4
#        x +   y       >= 1
#        x, y, z binary

import gurobipy as gp

with gp.Env() as env, gp.Model(env=env) as model:

    # Create variables
    x = model.addVar(vtype='B', name="x")
    y = model.addVar(vtype='B', name="y")
    z = model.addVar(vtype='B', name="z")

    # Set objective function
    model.setObjective(x + y + 2 * z, gp.GRB.MAXIMIZE)

    # Add constraints
    model.addConstr(x + 2 * y + 3 * z <= 4)
    model.addConstr(x + y >= 1)

    # Solve it!
    model.optimize()

    print(f"Optimal objective value: {model.ObjVal}")
    print(f"Solution values: x={x.X}, y={y.X}, z={z.X}")
"""

__version__ = "13.0.1"

from gurobipy._batch import Batch

from gurobipy._core import (
    Column,
    Constr,
    Env,
    GenConstr,
    GenExpr,
    LinExpr,
    QConstr,
    QuadExpr,
    SOS,
    Var,
    NLExpr,
    TempConstr,
    tuplelist,
    tupledict,
)

from gurobipy._core import abs_, all_, and_, any_, max_, min_, norm, or_

from gurobipy._core import (
    disposeDefaultEnv,
    getParamInfo,
    gurobi,
    paramHelp,
    read,
    readParams,
    resetParams,
    setParam,
    writeParams,
)

from gurobipy._exception import GurobiError

from gurobipy._grb import GRB

from gurobipy._helpers import multidict, quicksum

from gurobipy._matrixapi import (
    MConstr,
    MGenConstr,
    MLinExpr,
    MQConstr,
    MQuadExpr,
    MVar,
    MNLExpr,
    concatenate,
    hstack,
    vstack,
)

from gurobipy._model import Model

from gurobipy._load_model import loadModel

import gurobipy.nlfunc as nlfunc


# fmt: off
__all__ = [
    # _batch
    "Batch",
    # _core
    "Column", "Constr", "Env", "GenConstr", "GenExpr", "LinExpr", "QConstr",
    "QuadExpr", "SOS", "Var", "NLExpr", "TempConstr", "tuplelist", "tupledict",
    "abs_", "all_", "and_", "any_", "max_", "min_", "norm", "or_",
    "disposeDefaultEnv", "getParamInfo", "gurobi", "paramHelp", "read",
    "readParams", "resetParams", "setParam", "writeParams",
    # _exception
    "GurobiError",
    # _grb
    "GRB",
    # _helpers
    "multidict", "quicksum",
    # _matrixapi
    "MConstr", "MGenConstr", "MLinExpr", "MQConstr", "MQuadExpr", "MVar",
    "MNLExpr", "concatenate", "hstack", "vstack",
    # _model
    "Model",
    # _load_model
    "loadModel",
    # _nlfunc
    "nlfunc",
]
# fmt: on
