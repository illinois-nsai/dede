# Wrapper for 'GRB.Attr' object

class AttrConstClass(object):
  '''
  Attributes are used throughout the Gurobi interface to query and
  modify model properties.  You refer to them as you would any
  other object attribute.  For example, "print model.numConstrs"
  prints the number of constraints in a model.  You can assign new values to
  some attributes (e.g., model.ModelName = "New"), while others can only
  be queried.  Note that attribute modification is handled in a lazy fashion,
  so you won't see the effect of a change until after the next call to
  Model.update() or Model.optimize().

  Capitalization is ignored in Gurobi attribute names, so
  model.numConstrs and model.NumConstrs are equivalent.

  Gurobi attributes can be grouped into the following categories:

  General model attributes (e.g., model.numConstrs):

    numConstrs: Number of constraints
    numVars: Number of variables
    numSOS: Number of SOS constraints
    numQConstrs: Number of quadrtic constraints
    numGenConstrs: Number of general constraints
    numNZs: Number of non-zero coefficients
    dNumNZs: Number of non-zero coefficients (in double format)
    numQNZs: Number of non-zero quadratic objective coefficients
    numIntVars: Number of integer variables (including binary variables)
    numBinVars: Number of binary variables
    modelName: Model name
    modelSense: Model sense: minimization (1) or maximization (-1)
    objCon: Constant offset for objective function
    objVal: Objective value for current solution
    objBound: Best available lower bound on solution objective
    poolObjBound: Bound on objective for undiscovered solutions
    poolNObjVal: Retrieve the objective value of an alternate MIP solution
    poolObjVal: Deprecated since v13 - use poolNObjVal instead
    MIPGap: Current relative MIP optimality gap
    runtime: Runtime (in seconds) for most recent optimization
    work: Work (in work units) for most recent optimization
    Status: Current status of model (help(GRB) gives a list of codes)
    ConcurrentWinMethod: Winning method that solved the continuous problem with concurrent optimization
    solCount: Number of stored solutions
    iterCount: Number of simplex iterations performed
    nodeCount: Number of branch-and-cut nodes explored
    barIterCount: Number of barrier iterations performed
    nlbarIterCount: Number of NL barrier iterations performed
    PDHGIterCount: Number of PDHG iterations performed
    isMIP: Indicates whether the model is MIP (1) or not (0)
    isQP: Indicates whether the model has a quadratic objective
    isQCP: Indicates whether the model has quadratic constraints
    isMultiObj: Indicates whether the model has multiple objectives
    IISMinimal: Is computed IIS minimal?
    kappa: Condition number of current LP basis
    maxCoeff: Maximum constraint coefficient (in absolute value)
    minCoeff: Minimum non-zero constraint coefficient (in absolute value)
    maxBound: Maximum finite variable bound (in absolute value)
    minBound: Minimum non-zero variable bound (in absolute value)
    maxObjCoeff: Maximum objective coefficient (in absolute value)
    minObjCoeff: Minimum objective coefficient (in absolute value)
    maxRHS: Maximum linear constraint right-hand side (in absolute value)
    minRHS: Minimum linear constraint right-hand side (in absolute value)
    maxQCRHS: Maximum quadratic constraint right-hand side (in absolute value)
    minQCRHS: Minimum quadratic constraint right-hand side (in absolute value)
    maxQCCoeff: Maximum quadratic constraint coefficient in quadratic part (in absolute value)
    minQCCoeff: Minimum non-zero quadratic constraint coefficient in quadratic part (in absolute value)
    maxQCLCoeff: Maximum quadratic constraint coefficient in linear part (in absolute value)
    minQCLCoeff: Minimum non-zero quadratic constraint coefficient in linear part (in absolute value)
    maxQObjCoeff: Maximum quadratic objective coefficient (in absolute value)
    minQObjCoeff: Minimum quadratic objective coefficient (in absolute value)
    farkasProof: Infeasible amount for Farkas proof for infeasible models
    numStart: number of MIP starts
    fingerprint: fingerprint computed from the model data and attributes influencing the optimization process
    MemUsed: Current amount of memory allocated (in GB) in master environment
    MaxMemUsed: Maximum amount of memory allocated (in GB) in master environment

  Variable attributes (e.g., var.lb):

    lb: Lower bound
    ub: Upper bound
    obj: Objective coefficient
    vType: Variable type (GRB.CONTINUOUS, GRB.BINARY, GRB.INTEGER,
                          GRB.SEMICONT, or GRB.SEMIINT)
    varName: Variable name
    x: Solution value
    rc: Reduced cost
    poolNX: Solution value in alternate MIP solution
    xn: Deprecated since v13 - use poolNX instead
    start: Start vector (use GRB.UNDEFINED to leave a value unspecified)
    vBasis: Basis status
    varHintVal: Variable hint value
    varHintPri: Variable hint priority
    branchPriority: Variable branch priority
    partition: Variable partition
    poolIgnore: Ignore the variable in the solution identity check of the solution pool
    IISLB: Does LB participate in IIS? (for infeasible models)
    IISUB: Does UB participate in IIS? (for infeasible models)
    IISLBForce: Forces the lower bound to be in (1) the final IIS or to be not in (0) the final IIS (for infeasible models)
    IISUBForce: Forces the upper bound to be in (1) the final IIS or to be not in (0) the final IIS (for infeasible models)
    SAObjLow: Smallest objective coefficient for which basis remains optimal
    SAObjUp: Largest objective coefficient for which basis remains optimal
    SALBLow: Smallest lower bound for which basis remains optimal
    SALBUp: Largest lower bound for which basis remains optimal
    SAUBLow: Smallest upper bound for which basis remains optimal
    SAUBUp: Largest upper bound for which basis remains optimal
    unbdRay: Unbounded ray
    pStart: Primal solution (for warm-starting simplex)
    preFixVal: The value of the variable (for variables fixed by presolve)
    varPreStat: Status of variable in presolved model
    VTag: Tag string for variables (each defined variable tag MUST be unique)

  Constraint attributes (e.g., constr.sense):

    sense: Constraint sense
    rhs: Constraint right-hand side value
    constrName: Constraint name
    pi: Dual value
    slack: Constraint slack
    cBasis: Basis status
    lazy: Lazy constraint flag
    IISConstr: Does constraint participate in IIS? (for infeasible models)
    IISConstrForce: Forces the constraint to be in (1) the final IIS or to be not in (0) the final IIS (for infeasible models)
    SARHSLow: Smallest RHS value for which basis remains optimal
    SARHSUp: Largest RHS value for which basis remains optimal
    farkasDual: Farkas dual for infeasible models
    dStart: Dual solution (for warm-starting simplex)
    CTag: Tag string for constraints (each defined constraint tag MUST be unique)

  SOS (e.g., sos.IISSOS):

    IISSOS: Does SOS participate in IIS? (for infeasible models)
    IISSOSForce: Forces the SOS constraint to be in (1) the final IIS or to be not in (0) the final IIS (for infeasible models)

  Quadratic constraint attributes (e.g., qc.sense):

    QCsense: Quadratic constraint sense
    QCrhs: Quadratic constraint right-hand side value
    QCname: Quadratic constraint name
    IISQConstr: Does quadratic constraint participate in IIS? (for infeasible models)
    IISQConstrForce: Forces the quadratic constraint to be in (1) the final IIS or to be not in (0) the final IIS (for infeasible models)
    QCpi: Dual value
    QCslack: Quadratic constraint slack
    QCTag: Tag string for quadratic constraints (each defined
           quadratic constraint tag MUST be unique)

  GenConstr (e.g., genconstr.IISGenConstr):

    GenConstrType: General constraint type (e.g., GRB.GENCONSTR_MAX)
    GenConstrName: General constraint name
    IISGenConstr: Does general constraint participate in IIS? (for infeasible models)
    IISGenConstrForce: Forces the general constraint to be in (1) the final IIS or to be not in (0) the final IIS (for infeasible models)

  GenConstr (only for function constraints)

    FuncPieceError: error allowed for PWL translation
    FuncPieceLength: piece length for PWL translation
    FuncPieceRatio: control whether to link function values or to have
                    pieces below or above the function
    FuncPieces: control PWL translation whether to use equal piece length,
                to limit error or to limit the total number of pieces
    FuncNonlinear: control whether the function is treated as nonlinear or
                   as a PWL approximation


  Solution quality (e.g., model.constrVio):

    You generally access quality information through Model.printQuality().
    Please refer to the Attributes section of the Gurobi Reference Manual for
    the full list of quality attributes.

  Multi-objectives

    ObjN: objectives of multi-objectives
    ObjNCon: constant terms of multi-objectives
    ObjNPriority: priorities of multi-objectives
    ObjNWeight: weights of multi-objectives
    ObjNRelTol: relative tolerances of multi-objectives
    ObjNAbsTol: absolute tolerances of multi-objectives
    ObjNVal: objective value for multi-objectives
    ObjNName: names of multi-objectives
    NumObj: number of multi-objectives

    NumObjPasses: number of optimization passes during the multi-objective solve
    ObjNPass: optimization pass in which the selected objective function was solved
    ObjPassNIterCount: simplex iteration count for a pass during the multi-objective solve
    ObjPassNMIPGap: MIP gap for a pass during the multi-objective solve
    ObjPassNNodeCount: number of explored nodes for a pass during the multi-objective solve
    ObjPassNObjBound: objective bound for a pass during the multi-objective solve
    ObjPassNObjVal: objective value for a pass during the multi-objective solve
    ObjPassNOpenNodeCount: number of unexplored nodes for a pass during the multi-objective solve
    ObjPassNRuntime: run time for a pass during the multi-objective solve
    ObjPassNStatus: status for a pass during the multi-objective solve
    ObjPassNWork: work done for a pass during the multi-objective solve

  Multi-scenarios

    ScenNLB: modification to lower bound vector in current scenario
    ScenNUB: modification to upper bound vector in current scenario
    ScenNObj: modification to objective coefficient vector in current scenario
    ScenNRHS: modification to right hand side vector in current scenario
    ScenNName: name of current scenario
    ScenNX: value in current solution of current scenario
    ScenNObjBound: objective bound of current scenario
    ScenNObjVal: objective value of current solution of current scenario
    NumScenarios: number of scenarios in multi-scenario model

  Batch Objects

    BatchErrorCode: Last error code for this batch request
    BatchErrorMessage: Last error string for this batch request
    BatchID: String ID for this batch request
    BatchStatus: Current status of batch request (help(GRB) gives a list of codes)

  '''

  def __setattr__(self, name, value):
    raise AttributeError("Gurobi attribute constants are not modifiable")

# General model attributes

  NumConstrs      = "NumConstrs"
  NumVars         = "NumVars"
  NumSOS          = "NumSOS"
  NumQConstrs     = "NumQConstrs"
  NumGenConstrs   = "NumGenConstrs"
  NumNZs          = "NumNZs"
  DNumNZs         = "DNumNZs"
  NumQNZs         = "NumQNZs"
  NumQCNZs        = "NumQCNZs"
  NumIntVars      = "NumIntVars"
  NumBinVars      = "NumBinVars"
  NumPWLObjVars   = "NumPWLObjVars"
  ModelName       = "ModelName"
  ModelSense      = "ModelSense"
  ObjCon          = "ObjCon"
  IsMIP           = "IsMIP"
  IsQP            = "IsQP"
  IsQCP           = "IsQCP"
  IsMultiObj      = "IsMultiObj"
  IISMinimal      = "IISMinimal"
  Kappa           = "Kappa"
  KappaExact      = "KappaExact"
  MaxCoeff        = "MaxCoeff"
  MinCoeff        = "MinCoeff"
  MaxBound        = "MaxBound"
  MinBound        = "MinBound"
  MaxObjCoeff     = "MaxObjCoeff"
  MinObjCoeff     = "MinObjCoeff"
  MaxRHS          = "MaxRHS"
  MinRHS          = "MinRHS"
  MaxQCRHS        = "MaxQCRHS"
  MinQCRHS        = "MinQCRHS"
  MaxQCCoeff      = "MaxQCCoeff"
  MinQCCoeff      = "MinQCCoeff"
  MaxQCLCoeff     = "MaxQCLCoeff"
  MinQCLCoeff     = "MinQCLCoeff"
  MaxQObjCoeff    = "MaxQObjCoeff"
  MinQObjCoeff    = "MinQObjCoeff"
  ObjVal          = "ObjVal"
  ObjBound        = "ObjBound"
  ObjBoundC       = "ObjBoundC"
  PoolObjBound    = "PoolObjBound"
  PoolNObjVal     = "PoolNObjVal"
  PoolObjVal      = "PoolObjVal"
  MIPGap          = "MIPGap"
  Runtime         = "Runtime"
  Work            = "Work"
  Status          = "Status"
  SolCount        = "SolCount"
  IterCount       = "IterCount"
  NodeCount       = "NodeCount"
  BarStatus       = "BarStatus"
  BarIterCount    = "BarIterCount"
  NLBarIterCount  = "NLBarIterCount"
  PDHGIterCount   = "PDHGIterCount"
  FarkasProof     = "FarkasProof"
  NumStart        = "NumStart"
  TuneResultCount = "TuneResultCount"
  LicenseExpiration = "LicenseExpiration"
  Fingerprint     = "Fingerprint"
  ConcurrentWinMethod = "ConcurrentWinMethod"
  MemUsed         = "MemUsed"
  MaxMemUsed      = "MaxMemUsed"

# Variable attributes

  LB             = "LB"
  UB             = "UB"
  Obj            = "Obj"
  VType          = "VType"
  VarName        = "VarName"
  X              = "X"
  RC             = "RC"
  PoolNX         = "PoolNX"
  Xn             = "Xn"
  BarX           = "BarX"
  BarPi          = "BarPi"
  Start          = "Start"
  VarHintVal     = "VarHintVal"
  VarHintPri     = "VarHintPri"
  BranchPriority = "BranchPriority"
  Partition      = "Partition"
  VBasis         = "VBasis"
  PWLObjCvx      = "PWLObjCvx"
  PoolIgnore     = "PoolIgnore"
  IISLB          = "IISLB"
  IISUB          = "IISUB"
  IISLBForce     = "IISLBForce"
  IISUBForce     = "IISUBForce"
  SAObjLow       = "SAObjLow"
  SAObjUp        = "SAObjUp"
  SALBLow        = "SALBLow"
  SALBUp         = "SALBUp"
  SAUBLow        = "SAUBLow"
  SAUBUp         = "SAUBUp"
  UnbdRay        = "UnbdRay"
  PStart         = "PStart"
  PreFixVal      = "PreFixVal"
  VarPreStat     = "VarPreStat"
  VTag           = "VTag"

# Constraint attributes

  Sense      = "Sense"
  RHS        = "RHS"
  ConstrName = "ConstrName"
  Pi         = "Pi"
  Slack      = "Slack"
  CBasis     = "CBasis"
  IISConstr  = "IISConstr"
  IISConstrForce = "IISConstrForce"
  SARHSLow   = "SARHSLow"
  SARHSUp    = "SARHSUp"
  FarkasDual = "FarkasDual"
  DStart     = "DStart"
  Lazy       = "Lazy"
  CTag       = "CTag"

# SOS attributes

  IISSOS = "IISSOS"
  IISSOSForce = "IISSOSForce"

# Quadratic constraint attributes

  QCSense    = "QCSense"
  QCRHS      = "QCRHS"
  QCName     = "QCName"
  IISQConstr = "IISQConstr"
  IISQConstrForce = "IISQConstrForce"
  QCPi       = "QCPi"
  QCSlack    = "QCSlack"
  QCTag      = "QCTag"

# General constraint attributes

  GenConstrType = "GenConstrType"
  GenConstrName = "GenConstrName"
  IISGenConstr = "IISGenConstr"
  IISGenConstrForce = "IISGenConstrForce"

# General constraint attributes for functions

  FuncPieceError  = "FuncPieceError"
  FuncPieceLength = "FuncPieceLength"
  FuncPieceRatio  = "FuncPieceRatio"
  FuncPieces      = "FuncPieces"
  FuncNonlinear   = "FuncNonlinear"

# Quality attributes

  MaxVio               = "MaxVio"

  BoundVio             = "BoundVio"
  BoundSVio            = "BoundSVio"
  BoundVioIndex        = "BoundVioIndex"
  BoundSVioIndex       = "BoundSVioIndex"

  BoundVioSum          = "BoundVioSum"
  BoundSVioSum         = "BoundSVioSum"

  ConstrVio            = "ConstrVio"
  ConstrSVio           = "ConstrSVio"
  ConstrVioIndex       = "ConstrVioIndex"
  ConstrSVioIndex      = "ConstrSVioIndex"

  ConstrVioSum         = "ConstrVioSum"
  ConstrSVioSum        = "ConstrSVioSum"

  ConstrResidual       = "ConstrResidual"
  ConstrSResidual      = "ConstrSResidual"
  ConstrResidualIndex  = "ConstrResidualIndex"
  ConstrSResidualIndex = "ConstrSResidualIndex"

  ConstrResidualSum    = "ConstrResidualSum"
  ConstrSResidualSum   = "ConstrSResidualSum"

  DualVio              = "DualVio"
  DualSVio             = "DualSVio"
  DualVioIndex         = "DualVioIndex"
  DualSVioIndex        = "DualSVioIndex"

  DualVioSum           = "DualVioSum"
  DualSVioSum          = "DualSVioSum"

  DualResidual         = "DualResidual"
  DualSResidual        = "DualSResidual"
  DualResidualIndex    = "DualResidualIndex"
  DualSResidualIndex   = "DualSResidualIndex"

  DualResidualSum      = "DualResidualSum"
  DualSResidualSum     = "DualSResidualSum"

  ComplVio             = "ComplVio"
  ComplVioIndex        = "ComplVioIndex"
  ComplVioSum          = "ComplVioSum"

  IntVio               = "IntVio"
  IntVioIndex          = "IntVioIndex"
  IntVioSum            = "IntVioSum"

# Solution pool quality attributes

  PoolNMaxVio          = "PoolNMaxVio"
  PoolNBoundVio        = "PoolNBoundVio"
  PoolNBoundVioIndex   = "PoolNBoundVioIndex"
  PoolNBoundVioSum     = "PoolNBoundVioSum"
  PoolNConstrVio       = "PoolNConstrVio"
  PoolNConstrVioIndex  = "PoolNConstrVioIndex"
  PoolNConstrVioSum    = "PoolNConstrVioSum"
  PoolNIntVio          = "PoolNIntVio"
  PoolNIntVioIndex     = "PoolNIntVioIndex"
  PoolNIntVioSum       = "PoolNIntVioSum"

# Multi-objective attributes

  ObjN                 = "ObjN"
  ObjNCon              = "ObjNCon"
  ObjNPriority         = "ObjNPriority"
  ObjNWeight           = "ObjNWeight"
  ObjNRelTol           = "ObjNRelTol"
  ObjNAbsTol           = "ObjNAbsTol"
  ObjNVal              = "ObjNVal"
  ObjNName             = "ObjNName"
  NumObj               = "NumObj"

  NumObjPasses          = "NumObjPasses"
  ObjNPass              = "ObjNPass"
  ObjPassNIterCount     = "ObjPassNIterCount"
  ObjPassNMIPGap        = "ObjPassNMIPGap"
  ObjPassNNodeCount     = "ObjPassNNodeCount"
  ObjPassNObjBound      = "ObjPassNObjBound"
  ObjPassNObjVal        = "ObjPassNObjVal"
  ObjPassNOpenNodeCount = "ObjPassNOpenNodeCount"
  ObjPassNRuntime       = "ObjPassNRuntime"
  ObjPassNStatus        = "ObjPassNStatus"
  ObjPassNWork          = "ObjPassNWork"

# Multi-scenario attributes

  ScenNLB              = "ScenNLB"
  ScenNUB              = "ScenNUB"
  ScenNObj             = "ScenNObj"
  ScenNRHS             = "ScenNRHS"
  ScenNName            = "ScenNName"
  ScenNX               = "ScenNX"
  ScenNObjBound        = "ScenNObjBound"
  ScenNObjVal          = "ScenNObjVal"
  NumScenarios         = "NumScenarios"

# Batch attributes

  BatchErrorCode       = "BatchErrorCode"
  BatchErrorMessage    = "BatchErrorMessage"
  BatchID              = "BatchID"
  BatchStatus          = "BatchStatus"
