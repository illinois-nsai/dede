# Wrapper for 'GRB.Param' object

class ParamConstClass(object):
  '''
  Gurobi parameters are used to control the optimization process.  They all
  have default values, but their values can be changed using the setParam()
  function.  Current values can be retrieved using the Model.getParamInfo()
  method.

  Parameters fall into the following categories:

  Termination: affect the termination of an optimize() call
    BarIterLimit: limits the number of barrier iterations performed
    BestBdStop: sets a best bound values at which optimization should stop
    BestObjStop: sets an objective value at which optimization should stop
    Cutoff: sets a target objective value
    IterationLimit: limits the number of simplex iterations performed
    MemLimit: returns an error if the total amount of memory used by Gurobi exceeds this limit (in GB)
    NLBarIterLimit: limits the number of NL barrier iterations performed
    NodeLimit: limits the number of MIP nodes explored
    PDHGIterLimit: limits the number of PDHG iterations performed
    SoftMemLimit: limits the total amount of memory that Gurobi can use (in GB)
    SolutionLimit: sets a target for the number of feasible solutions found
    TimeLimit: limits the total time expended (in seconds)
    WorkLimit: limits the total work expended (in work units)

  Tolerances: control the allowable feasibility or optimality violations
    BarConvTol: barrier convergence tolerance
    BarQCPConvTol: barrier convergence tolerance for QCP models
    FeasibilityTol: primal feasibility tolerance
    IntFeasTol: integer feasibility tolerance
    MarkowitzTol: threshold pivoting tolerance
    MIPGap: target relative MIP optimality gap
    MIPGapAbs: target absolute MIP optimality gap
    NLBarCFeasTol: NL barrier complementarity feasibility tolerance
    NLBarDFeasTol: NL barrier dual feasibility tolerance
    NLBarPFeasTol: NL barrier primal feasibility tolerance
    OptimalityTol: dual feasibility tolerance
    PDHGRelTol: PDHG relative termination tolerance
    PDHGAbsTol: PDHG absolute termination tolerance
    PDHGConvTol: PDHG relative objective gap tolerance
    PSDTol: QP positive semidefinite tolerance

  Simplex: affect the simplex algorithms
    InfUnbdInfo: makes additional information available for infeasible or
                 unbounded LP models
    LPWarmStart: controls whether and how to warm-start LP optimization
    NetworkAlg: controls whether to use network simplex, if an LP is
                a network problem
    NormAdjust: chooses different pricing norm variants
    ObjScale: controls objective scaling
    PerturbValue: controls the magnitude of any simplex perturbations
    Quad: turns quad precision on or off
    ScaleFlag: turns model scaling on or off
    Sifting: dual simplex sifting strategy for LP, MIP root and MIP nodes
    SiftMethod: chooses from dual, primal and barrier to solve sifting
                subproblems
    SimplexPricing: determines variable pricing strategy
    SolutionTarget: specifies the solution target for LPs

  Barrier: affect the barrier algorithms
    BarCorrectors: limits the number of central corrections
    BarHomogeneous: selects the barrier homogeneous algorithm
    BarOrder: determines the fill reducing ordering strategy
    Crossover: controls barrier crossover
    CrossoverBasis: controls initial crossover basis construction
    QCPDual: enables dual variable computation for continuous QCP models

  MIP: affect the MIP algorithms
    BranchDir: controls the branching node selection
    ConcurrentMIP: enables concurrent MIP optimization
    ConcurrentJobs: enables distributed concurrent optimization
    DegenMoves: limit degenerate simplex moves
    FixVarsInIndicators: controls whether variables in indicators should be fixed in a fixed model
    Heuristics: controls the amount of time spent in MIP heuristics
    ImproveStartGap: gap at which to switch MIP search strategies
    ImproveStartNodes: node count at which to switch MIP search strategies
    ImproveStartTime: time at which to switch MIP search strategies
    ImproveStartWork: work at which to switch MIP search strategies
    IntegralityFocus: controls integrality focus
    MinRelNodes: controls the minimum relaxation heuristic
    MIPFocus: affects the high-level MIP search strategy
    MIQCPMethod: controls whether to solve QCP node relaxation or to use OA
    NLPHeur: controls the NLP heuristic for non-convex quadratic models
    NodefileDir: determines the directory used to store nodes on disk
    NodefileStart: memory nodes may use (in GB) before being written to disk
    NodeMethod: determines the algorithm used to solve MIP node relaxations
    NonConvex: controls how to deal with non-convex quadratic programs
    NoRelHeurSolutions: limits the number of solutions found by the NoRel heuristic
    NoRelHeurTime: controls the time spent in the NoRel heuristic
    NoRelHeurWork: controls the work spent in the NoRel heuristic
    OBBT: controls the aggressiveness of optimality-based bound tightening
    PartitionPlace: controls when the partition heuristic runs
    PumpPasses: controls the feasibility pump heuristic
    RINS: sets the frequency of the RINS heuristic
    SolFiles: location to store intermediate solution files
    SolutionNumber: controls access to alternate MIP solutions
    StartNodeLimit: limits nodes in MIP start sub-MIP
    StartNumber: selects the MIP start index
    StartTimeLimit: limits time (in seconds) in MIP start sub-MIP
    StartWorkLimit: limits work (in work units) in MIP start sub-MIP
    SubMIPNodes: limits the numbers of nodes explored in a RINS sub-MIP
    Symmetry: controls access to alternate MIP solutions
    VarBranch: controls the branch variable selection strategy
    ZeroObjNodes: controls the zero objective heuristic

  Presolve: affect the presolve algorithms
    AggFill: controls the level of presolve aggregation
    Aggregate: turns presolve aggregation on or off
    DualReductions: controls presolve dual reductions
    PreCrush: allows presolve to crush any user cut
    PreDepRow: controls the presolve dependent row reduction
    PreDual: determines whether presolve forms the dual of the input model
    PreMIQCPForm: chooses the form for MIQCP presolved model
    PrePasses: limits the number of presolve passes
    PreQLinearize: controls presolve Q matrix linearization
    Presolve: turns presolve on or off
    PreSOS1BigM: threshold for presolve SOS1 conversion to binary form
    PreSOS1Encoding: Encoding used to reformulate SOS1
    PreSOS2BigM: threshold for presolve SOS2 conversion to binary form
    PreSOS2Encoding: Encoding used to reformulate SOS2
    PreSparsify: enables the presolve sparsify reduction

  Tuning: affect the operation of the tuning tool
    TuneCriterion: specify different tuning criteria
    TuneJobs: enables distributed tuning using permanent workers
    TuneDynamicJobs: enables distributed tuning using dynamic workers
    TuneOutput: tuning output level
    TuneResults: number of imroved parameter sets returned
    TuneTimeLimit: tuning time limit
    TuneTrials: number of trial runs with each parameter set
    TuneCleanup: time percentage spend in a cleanup phase
    TuneTargetMIPGap: a target gap to be reached
    TuneTargetTime: a target runtime in seconds to be reached
    TuneMetric: method for aggregating results

  Multiple solutions: determines how the MIP search looks for solutions
    PoolGap: determines the quality of the retained solutions (relative)
    PoolGapAbs: determines the quality of the retained solutions (absolute)
    PoolSearchMode: chooses the approach used to search for solutions
    PoolSolutions: determines the number of solutions that are stored

  MIP cuts: affect the generation of MIP cutting planes
    BQPCuts: controls BQP cut generation
    CliqueCuts: controls clique cut generation
    CoverCuts: controls cover cut generation
    CutAggPasses: limits aggregation during cut generation
    CutPasses: limits the number of cut passes
    Cuts: global cut generation control
    DualImpliedCuts: controls dual implied bound cut generation
    FlowCoverCuts: controls flow cover cut generation
    FlowPathCuts: controls flow path cut generation
    GomoryPasses: controls the number of Gomory cut passes
    GUBCoverCuts: controls GUB cover cut generation
    ImpliedCuts: controls implied bound cut generation
    InfProofCuts: controls infeasibility proof cut generation
    MIPSepCuts: controls MIP separation cut generation
    MIRCuts: controls MIR cut generation
    ModKCuts: controls mod-k cut generation
    NetworkCuts: controls network cut generation
    ProjImpliedCuts: controls projected implied bound cut generation
    PSDCuts: controls PSD cut generation
    LiftProjectCuts: controls lift-and-project cut generation
    MasterKnapsackCuts: controls master knapsack cut generation
    MixingCuts: controls mixing cut generation
    RelaxLiftCuts: controls relax-and-lift cut generation
    RLTCuts: controls RLT cut generation
    StrongCGCuts: controls Strong-CG cut generation
    SubMIPCuts: controls sub-MIP cut generation
    ZeroHalfCuts: controls zero-half cut generation

  NL barrier: affect the NL barrier algorithm
    OptimalityTarget: selects search for global or local optimum

  Distributed algorithms: used for distributed optimization
    WorkerPassword: cluster client password
    WorkerPool: server URL to access the cluster

  Cloud: parameters used for cloud-based optimization
    CloudAccessID: Instant Cloud access ID
    CloudPool: Instant Cloud pool name
    CloudSecretKey: Instant Cloud secret key

  Compute Server and Cluster Manager: used for optimization with Remote Services
    CSAPIAccessID: API access ID to access the Cluster Manager
    CSAPISecret: API secret key to access the Cluster Manager
    CSAppName: application name
    CSAuthToken: Authentication token used internally to access a Cluster Manager
    CSBatchMode: Controls Batch-Mode optimization with a Cluster Manager
    CSClientLog: Turns logging on or off
    CSGroup: Group placement request for cluster
    CSIdleTimeout: job idle timeout
    CSManager: access URL of the Cluster Manager
    CSPriority: compute server job priority
    CSQueueTimeout: queue timeout for new jobs
    CSRouter: remote services router URL
    CSTLSInsecure: enable TLS insecurity mode
    ComputeServer: server URL to access the cluster
    ServerPassword: cluster client password
    ServerTimeout: network timeout
    UserName: User name to use when connecting to the Cluster Manager

  Token Server: affect token server parameters
    TokenServer: address of token server
    TSPort: token server port

  Web License Service (WLS): affect WLS parameters
    WLSAccessID: WLS access ID
    WLSSecret: WLS secret key
    WLSTokenDuration: lifespan of a Web License Services token (in minutes)
    WLSTokenRefresh: Relative refresh interval for Web License Services tokens
    WLSToken: WLS token
    LicenseID: WLS license ID
    WLSProxy: WLS proxy
    WLSConfig: WLS configuration

  Other:
    Disconnected: controls the disconnected component strategy
    DisplayInterval: sets the frequency at which log lines are printed
    FeasRelaxBigM: BigM value for feasibility relaxation
    FuncMaxVal: maximal value for |lb| and |ub| of x and y variables of
                general function constraints
    FuncNonlinear: controls whether nonlinear general constraints are treated
                   as nonlinear constraints or whether they are approximated
                   via piece-wise linear constraints
    FuncPieceError: error allowed for PWL translation of general function
                    constraints without own options specified
    FuncPieceLength: piece length for PWL translation of general function
                     constraints without own options specified
    FuncPieceRatio: control whether to link function values or to have
                    pieces below or above the function
    FuncPieces: control PWL translation of general function constraints
                without own options specified, whether to use equal piece
                length, to limit error or to limit the total number of pieces
    IgnoreNames: indicates whether to ignore names provided by users
    IISMethod: method used to find an IIS
    InheritParams: control how a supporting environment is applied
    JSONSolDetail: controls amount of information in a JSON solution string
    LazyConstraints: programs that use lazy constraints must set this to 1
    LogFile: sets the name of the Gurobi log file
    LogToConsole: turn logging to the console on or off
    Method: algorithm used to solve a continuous model or the root node of a
            MIP model (auto, primal simplex, dual simplex, barrier, or
            concurrent)
    ConcurrentMethod: algorithms used with concurrently when solving continuous
                      models using concurrent
    NumericFocus: controls numerically conservative level
    MultiObjMethod: warm-start method to solve for subsequent objectives
    MultiObjPre: controls initial presolve level on multi-objective models
    ObjNumber: selects the objective index of multi-objectives
    ObjPassNumber: selects the optimiaztion pass index of multi-objective solve
    OutputFlag: turn logging on or off
    PDHGGPU: Enable GPU for PDHG
    Record: enables replay
    ResultFile: result file to write when optimization completes
    ScenarioNumber: selects the scenario index of multi-scenario models
    Seed: sets the random number seed
    ThreadLimit: sets a thread limit when starting an environment
    Threads: sets the number of threads to apply to parallel MIP
    UpdateMode: controls the way how to update a model

  Parameters can be referred to using the Param class (e.g.
  "setParam(GRB.param.threads, 1)"), or by using the name as a string
  (e.g., "setParam('threads', 1)).  You can use the '*' and '?' wildcards
  when inputting parameter names, and text case is ignored
  (so "setParam('thr*', 1)" would also work).

  For further information on any of these parameters, type
  paramHelp('paramname') (e.g., paramHelp("NodeLimit")).  Wildcards
  are also accepted for paramHelp().
  '''

  def __setattr__(self, name, value):
    raise AttributeError("Gurobi parameter constants are not modifiable")

  BarIterLimit   = "BarIterLimit"
  Cutoff         = "Cutoff"
  IterationLimit = "IterationLimit"
  MemLimit       = "MemLimit"
  NLBarIterLimit = "NLBarIterLimit"
  NodeLimit      = "NodeLimit"
  SoftMemLimit   = "SoftMemLimit"
  SolutionLimit  = "SolutionLimit"
  TimeLimit      = "TimeLimit"
  WorkLimit      = "WorkLimit"
  BestObjStop    = "BestObjStop"
  BestBdStop     = "BestBdStop"

  BarConvTol     = "BarConvTol"
  BarQCPConvTol  = "BarQCPConvTol"
  FeasibilityTol = "FeasibilityTol"
  IntFeasTol     = "IntFeasTol"
  MarkowitzTol   = "MarkowitzTol"
  MIPGap         = "MIPGap"
  MIPGapAbs      = "MIPGapAbs"
  NLBarCFeasTol  = "NLBarCFeasTol"
  NLBarDFeasTol  = "NLBarDFeasTol"
  NLBarPFeasTol  = "NLBarPFeasTol"
  OptimalityTol  = "OptimalityTol"
  PSDTol         = "PSDTol"

  InfUnbdInfo    = "InfUnbdInfo"
  LPWarmStart    = "LPWarmStart"
  NetworkAlg     = "NetworkAlg"
  NormAdjust     = "NormAdjust"
  ObjScale       = "ObjScale"
  PerturbValue   = "PerturbValue"
  Quad           = "Quad"
  ScaleFlag      = "ScaleFlag"
  Sifting        = "Sifting"
  SiftMethod     = "SiftMethod"
  SimplexPricing = "SimplexPricing"
  SolutionTarget = "SolutionTarget"

  BarCorrectors  = "BarCorrectors"
  BarHomogeneous = "BarHomogeneous"
  BarOrder       = "BarOrder"
  Crossover      = "Crossover"
  CrossoverBasis = "CrossoverBasis"
  QCPDual        = "QCPDual"

  PDHGIterLimit = "PDHGIterLimit"
  PDHGRelTol    = "PDHGRelTol"
  PDHGAbsTol    = "PDHGAbsTol"
  PDHGConvTol   = "PDHGConvTol"
  PDHGGPU       = "PDHGGPU"

  BranchDir           = "BranchDir"
  DegenMoves          = "DegenMoves"
  ConcurrentJobs      = "ConcurrentJobs"
  ConcurrentMIP       = "ConcurrentMIP"
  Disconnected        = "Disconnected"
  DistributedMIPJobs  = "DistributedMIPJobs"
  FixVarsInIndicators = "FixVarsInIndicators"
  Heuristics          = "Heuristics"
  ImproveStartGap     = "ImproveStartGap"
  ImproveStartNodes   = "ImproveStartNodes"
  ImproveStartTime    = "ImproveStartTime"
  ImproveStartWork    = "ImproveStartWork"
  MinRelNodes         = "MinRelNodes"
  MIPFocus            = "MIPFocus"
  MIQCPMethod         = "MIQCPMethod"
  NLPHeur             = "NLPHeur"
  NodefileDir         = "NodefileDir"
  NodefileStart       = "NodefileStart"
  NodeMethod          = "NodeMethod"
  NonConvex           = "NonConvex"
  NoRelHeurSolutions  = "NoRelHeurSolutions"
  NoRelHeurTime       = "NoRelHeurTime"
  NoRelHeurWork       = "NoRelHeurWork"
  OBBT                = "OBBT"
  PartitionPlace      = "PartitionPlace"
  PumpPasses          = "PumpPasses"
  RINS                = "RINS"
  SolFiles            = "SolFiles"
  SolutionNumber      = "SolutionNumber"
  SubMIPNodes         = "SubMIPNodes"
  Symmetry            = "Symmetry"
  VarBranch           = "VarBranch"
  ZeroObjNodes        = "ZeroObjNodes"

  TuneCriterion     = "TuneCriterion"
  TuneJobs          = "TuneJobs"
  TuneDynamicJobs   = "TuneDynamicJobs"
  TuneMetric        = "TuneMetric"
  TuneOutput        = "TuneOutput"
  TuneResults       = "TuneResults"
  TuneTimeLimit     = "TuneTimeLimit"
  TuneTrials        = "TuneTrials"
  TuneCleanup       = "TuneCleanup"
  TuneTargetMIPGap  = "TuneTargetMIPGap"
  TuneTargetTime    = "TuneTargetTime"

  PoolSearchMode = "PoolSearchMode"
  PoolSolutions  = "PoolSolutions"
  PoolGap        = "PoolGap"
  PoolGapAbs     = "PoolGapAbs"

  BQPCuts         = "BQPCuts"
  Cuts            = "Cuts"
  CliqueCuts      = "CliqueCuts"
  CoverCuts       = "CoverCuts"
  FlowCoverCuts   = "FlowCoverCuts"
  FlowPathCuts    = "FlowPathCuts"
  GUBCoverCuts    = "GUBCoverCuts"
  ImpliedCuts     = "ImpliedCuts"
  InfProofCuts    = "InfProofCuts"
  MIPSepCuts      = "MIPSepCuts"
  MIRCuts         = "MIRCuts"
  ModKCuts        = "ModKCuts"
  NetworkCuts     = "NetworkCuts"
  DualImpliedCuts = "DualImpliedCuts"
  ProjImpliedCuts = "ProjImpliedCuts"
  PSDCuts         = "PSDCuts"
  LiftProjectCuts = "LiftProjectCuts"
  MasterKnapsackCuts = "MasterKnapsackCuts"
  MixingCuts      = "MixingCuts"
  RelaxLiftCuts   = "RelaxLiftCuts"
  RLTCuts         = "RLTCuts"
  StrongCGCuts    = "StrongCGCuts"
  SubMIPCuts      = "SubMIPCuts"
  ZeroHalfCuts    = "ZeroHalfCuts"
  CutAggPasses    = "CutAggPasses"
  CutPasses       = "CutPasses"
  GomoryPasses    = "GomoryPasses"

  OptimalityTarget = "OptimalityTarget"

  WorkerPassword   = "WorkerPassword"
  WorkerPool       = "WorkerPool"
  ComputeServer    = "ComputeServer"
  ServerPassword   = "ServerPassword"
  ServerTimeout    = "ServerTimeout"
  CSRouter         = "CSRouter"
  CSGroup          = "CSGroup"
  CSPriority       = "CSPriority"
  CSQueueTimeout   = "CSQueueTimeout"
  CSTLSInsecure    = "CSTLSInsecure"
  CSAppName        = "CSAppName"
  CSClientLog      = "CSClientLog"
  CSIdleTimeout    = "CSIdleTimeout"
  TokenServer      = "TokenServer"
  TSPort           = "TSPort"
  CloudAccessID    = "CloudAccessID"
  CloudSecretKey   = "CloudSecretKey"
  CloudPool        = "CloudPool"
  CloudHost        = "CloudHost"
  JobID            = "JobID"
  CSAPIAccessID    = "CSAPIAccessID"
  CSAPISecret      = "CSAPISecret"
  CSAuthToken      = "CSAuthToken"
  CSBatchMode      = "CSBatchMode"
  CSManager        = "CSManager"
  UserName         = "UserName"
  WLSAccessID      = "WLSAccessID"
  WLSSecret        = "WLSSecret"
  WLSTokenDuration = "WLSTokenDuration"
  WLSTokenRefresh  = "WLSTokenRefresh"
  WLSToken         = "WLSToken"
  LicenseID        = "LicenseID"
  WLSProxy         = "WLSProxy"
  WLSConfig        = "WLSConfig"

  AggFill         = "AggFill"
  Aggregate       = "Aggregate"
  DisplayInterval = "DisplayInterval"
  DualReductions  = "DualReductions"
  FeasRelaxBigM   = "FeasRelaxBigM"
  FuncMaxVal      = "FuncMaxVal"
  FuncNonlinear   = "FuncNonlinear"
  FuncPieceError  = "FuncPieceError"
  FuncPieceLength = "FuncPieceLength"
  FuncPieceRatio  = "FuncPieceRatio"
  FuncPieces      = "FuncPieces"
  IISMethod       = "IISMethod"
  IntegralityFocus = "IntegralityFocus"
  InheritParams   = "InheritParams"
  JSONSolDetail   = "JSONSolDetail"
  LazyConstraints = "LazyConstraints"
  LogFile         = "LogFile"
  LogToConsole    = "LogToConsole"
  Method          = "Method"
  ConcurrentMethod = "ConcurrentMethod"
  NumericFocus    = "NumericFocus"
  MultiObjMethod  = "MultiObjMethod"
  MultiObjPre     = "MultiObjPre"
  IgnoreNames     = "IgnoreNames"
  ObjNumber       = "ObjNumber"
  ObjPassNumber   = "ObjPassNumber"
  OutputFlag      = "OutputFlag"
  PreCrush        = "PreCrush"
  PreDepRow       = "PreDepRow"
  PreDual         = "PreDual"
  PrePasses       = "PrePasses"
  PreQLinearize   = "PreQLinearize"
  Presolve        = "Presolve"
  PreSOS1BigM     = "PreSOS1BigM"
  PreSOS1Encoding = "PreSOS1Encoding"
  PreSOS2BigM     = "PreSOS2BigM"
  PreSOS2Encoding = "PreSOS2Encoding"
  PreSparsify     = "PreSparsify"
  PreMIQCPForm    = "PreMIQCPForm"
  Record          = "Record"
  ResultFile      = "ResultFile"
  ScenarioNumber  = "ScenarioNumber"
  Seed            = "Seed"
  StartNodeLimit  = "StartNodeLimit"
  StartNumber     = "StartNumber"
  StartTimeLimit  = "StartTimeLimit"
  StartWorkLimit  = "StartWorkLimit"
  ThreadLimit     = "ThreadLimit"
  Threads         = "Threads"
  UpdateMode      = "UpdateMode"
