"""gurobipy nonlinear functions module

This module contains a number of functions for creating nonlinear expressions.
Each function can be called with any modeling object and returns an "NLExpr" or
"MNLExpr" representing the corresponding nonlinear expression.  The resulting
object can be used to add nonlinear constraints to the model. For example::

   import gurobipy as gp
   from gurobipy import GRB, nlfunc

   with gp.Env() as env, gp.Model(env=env) as model:

       x = model.addVar(lb=-GRB.INFINITY, name="x")
       y = model.addVar(lb=-GRB.INFINITY, name="y")
       z = model.addVar(lb=-GRB.INFINITY, name="z")

       # Create a constraint specifying z = sin(x + y)
       model.addConstr(z == nlfunc.sin(x + y))
"""

from gurobipy._helpers import (
    sqrt,
    sin,
    cos,
    tan,
    exp,
    log,
    log2,
    log10,
    logistic,
    signpow,
    square,
    tanh,
)
