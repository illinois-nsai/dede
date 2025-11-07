from gurobipy._grb import GRB
from gurobipy._lowlevel import _load_model_memoryview, _add_qp_terms_memoryview
from gurobipy._util import _isscalar


def _double_array_argument(arg, size):
    """If arg is a single value, build a repeating array. Otherwise,
    let numpy handle conversion from an arbitrary iterator."""
    import numpy as np

    if _isscalar(arg):
        arg = float(arg)
        repeat = np.empty(size, dtype="float64")
        repeat.fill(arg)
        return repeat

    else:
        return np.require(arg, dtype="float64", requirements=["C"])


def _char_array_argument(arg, size):
    """Convert str or bytearray to bytes. If a single value is given and
    size is not None, build a repeating byte array. Otherwise, let numpy
    handle conversion from an arbitrary iterator."""

    import numpy as np

    # Convert to bytes first if possible. Single character case still
    # needs to be handled after conversion.
    if isinstance(arg, str):
        arg = arg.encode()

    if isinstance(arg, (bytes, bytearray)):
        if size is not None and len(arg) == 1:
            return arg * size
        else:
            return arg
    else:
        return np.require(arg, dtype="S1", requirements=["C"]).tobytes()


def loadModel(
    *,
    env,
    numvars,
    numconstrs,
    modelsense=GRB.MINIMIZE,
    objcon=0.0,
    obj=None,
    lb=None,
    ub=None,
    vtype=None,
    constr_csc=None,
    sense=None,
    rhs=None,
    qobj_coo=None,
    name=None,
):
    """
    ROUTINE:
      loadModel(
        *, env, numvars, numconstrs, modelsense=GRB.MINIMIZE, objcon=0.0,
        obj=None, lb=None, ub=None, vtype=None, constr_csc=None, sense=None,
        rhs=None, qobj_coo=None, name=None
      )

    PURPOSE:
      Create a new optimization model, using the provided arguments to
      initialize the model data (objective function, variable bounds,
      constraint matrix, etc.). The model is then ready for optimization.

      This function allows you to build models with only linear constraints,
      and linear or quadratic objectives.

      You’ll need "numpy" installed to use this function. All arguments to
      this function are keyword-only.

    ARGUMENTS:
      env – The environment in which the new model should be created.
      numvars – The number of variables in the model.
      numconstrs – The number of constraints in the model.
      modelsense – The sense of the objective function. Allowed
        values are "GRB.MINIMIZE" or "GRB.MAXIMIZE". Defaults to
        "GRB.MINIMIZE".
      objcon – Constant objective offset (defaults to "0.0").
      obj – (optional) Objective coefficients for the model
        variables, as a list or 1-D array of length "numvars". If not
        provided, all linear objective coefficients in the model are set
        to "0.0".
      lb – (optional) Lower bounds for the model variables, as a
        list or 1-D array of length "numvars". If not provided, all
        variables will have lower bounds of "0.0".
      ub – (optional) Upper bounds for the model variables, as a
        list or 1-D array of length "numvars". If not provided, all
        variables will have infinite upper bounds.
      vtype – (optional) Types for the variables, as a list or 1-D
        array of length "numvars". Options are "GRB.CONTINUOUS",
        "GRB.BINARY", "GRB.INTEGER", "GRB.SEMICONT", or "GRB.SEMIINT". If
        not provided, all variables will be continuous.
      constr_csc – (optional) Linear constraint data in Compressed
        Sparse Column format (CSC) as a tuple "(data, indices, indptr)".
        In this format the constraint indices for variable i are stored
        in "indices[indptr[i]:indptr[i+1]]" and their corresponding
        coefficients are stored in "data[indptr[i]:indptr[i+1]]". The
        format is the same as that used by "scipy.sparse.csc_array". This
        argument can be omitted if no linear constraints are being added.
      sense – (optional) The senses for the model constraints, as a
        list or 1-D array of length "numconstrs". Options are
        "GRB.EQUAL", "GRB.LESS_EQUAL", or "GRB.GREATER_EQUAL". Can be
        omitted if no linear constraints are being added.
      rhs – (optional) Right-hand side values for the new
        constraints, as a list or 1-D array of length "numconstrs". Can
        be omitted if no linear constraints are being added.
      qobj_coo – (optional) Quadratic objective matrix in
        coordinate format as a tuple "(qval, (qrow, qcol))". The i^{th}
        quadratic term is represented using three values: a pair of
        indices (stored in "qrow[i]" and "qcol[i]"), and a coefficient
        (stored in "qval[i]"). The format is the same as that used by
        "scipy.sparse.coo_array". This argument can be omitted if no
        quadratic objective terms are being added.
      name – (optional) The name of the model.

    RETURN VALUE:
      A Model object.
    """

    import numpy as np

    if name is None:
        modelname = b""
    elif isinstance(name, str):
        modelname = name.encode()
    else:
        raise TypeError("'modelname' must be a string or None")

    try:
        numvars = int(numvars)
    except ValueError as e:
        raise ValueError("'numvars' must be an integer") from e

    try:
        numconstrs = int(numconstrs)
    except ValueError as e:
        raise ValueError("'numconstrs' must be an integer") from e

    if numvars < 0:
        raise ValueError("'numvars' must be non-negative")
    if numconstrs < 0:
        raise ValueError("'numconstrs' must be non-negative")

    try:
        modelsense = int(modelsense)
    except ValueError as e:
        raise ValueError("'modelsense' must be an integer") from e

    try:
        objcon = float(objcon)
    except ValueError as e:
        raise ValueError("'objcon' must be numeric") from e

    # Prepare variable attribute arrays, repeating scalars if needed
    if obj is not None:
        obj = _double_array_argument(obj, size=numvars)
    if lb is not None:
        lb = _double_array_argument(lb, size=numvars)
    if ub is not None:
        ub = _double_array_argument(ub, size=numvars)
    if vtype is not None:
        vtype = _char_array_argument(vtype, size=numvars)

    if numconstrs == 0:
        # No linear constraint matrix was given; construct an empty one
        rhs = np.array([], dtype="float64")
        sense = b""
        vbeg = np.zeros((numvars,), dtype="uint64")
        vlen = np.zeros((numvars,), dtype="int32")
        vind = np.array([], dtype="int32")
        vval = np.array([], dtype="float64")

    else:
        # To avoid ambiguity, a string for 'sense' must have length numconstrs.
        # no '<=' or '<' allowed to set the same sense for all constraints.
        sense = _char_array_argument(sense, size=None)

        rhs = _double_array_argument(rhs, size=numconstrs)

        try:
            data, indices, indptr = constr_csc
        except Exception as e:
            raise ValueError(
                "'constr_csc' must be a tuple of the form (data, indices, indptr)"
            ) from e

        vind = np.require(indices, dtype="int32", requirements=["C"])
        vval = np.require(data, dtype="float64", requirements=["C"])

        indptr = np.require(indptr, dtype="uint64", requirements=["C"])
        vlen = np.empty(len(indptr) - 1, dtype="int32")
        np.subtract(indptr[1:], indptr[:-1], out=vlen)
        vbeg = indptr[:-1]

    model = _load_model_memoryview(
        env=env,
        numvars=numvars,
        numconstrs=numconstrs,
        modelname=modelname,
        objsense=modelsense,
        objcon=objcon,
        obj=obj,
        sense=sense,
        rhs=rhs,
        vbeg=vbeg,
        vlen=vlen,
        vind=vind,
        vval=vval,
        lb=lb,
        ub=ub,
        vtype=vtype,
    )

    # Any further operations must be followed by an update() call. The model
    # must be cleaned up if an exception is raised after this point.

    if qobj_coo is not None:

        try:
            qval, (qrow, qcol) = qobj_coo
        except Exception as e:
            model.close()
            raise ValueError(
                "'qobj_coo' must be a tuple of the form (qval, (qrow, qcol))"
            ) from e

        try:
            numqnz = len(qrow)
            _add_qp_terms_memoryview(
                model=model,
                numqnz=numqnz,
                qrow=np.require(qrow, dtype="int32", requirements=["C"]),
                qcol=np.require(qcol, dtype="int32", requirements=["C"]),
                qval=np.require(qval, dtype="float64", requirements=["C"]),
            )
            model.update()
        except Exception:
            model.close()
            raise

    return model
