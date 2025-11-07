# Wrapper for 'GRB.Callback' object

class CallbackConstClass(object):
  '''
  Callbacks are user methods that are called by the Gurobi solver
  during the optimization.  To use a callback, define a method
  that takes two integer arguments (model and where), and pass it
  as the argument to Model.optimize.  Once optimization begins,
  your method will be called with one of the following 'where' values:

  Possible 'where' values (e.g., where == GRB.Callback.MIP) are:

    POLLING:  Regular polling callback - no user queries allowed
    PRESOLVE: In presolve
    SIMPLEX:  In simplex
    BARRIER:  In barrier
    MIP:      In MIP
    MIPSOL:   New MIP incumbent available
    MIPNODE:  MIP node information available
    MULTIOBJ: In multi-objective optimization
    IIS:      In IIS computation
    NLBAR:    In NL barrier
    MESSAGE:  Optimizer output a message

  Your method can call Model.cbGet() to obtain detailed information
  on the progress of the optimization.  Allowed values depend
  on 'where'.  The prefix of the 'what' name indicate which
  ones are allowed for each 'where' (so 'PRE_COLDEL' can only
  be called when where == SIMPLEX).

  Allowed 'what' values (e.g., cbGet(GRB.Callback.MIP_OBJBND)) are:

    RUNTIME: Elapsed solver runtime
    WORK: Elapsed solver work
    MEMUSED: Current size of allocated memory (in GB)
    MAXMEMUSED: Maximum size of allocated memory (in GB)
    PRE_COLDEL: Deleted column count
    PRE_ROWDEL: Deleted row count
    PRE_SENCHG: Changed constraint sense count
    PRE_BNDCHG: Bound change count
    SPX_ITRCNT: Iteration count
    SPX_OBJVAL: Primal objective value
    SPX_PRIMINF: Primal infeasibility
    SPX_DUALINF: Dual infeasibility
    SPX_ISPERT: Has model been perturbed?
    BARRIER_ITRCNT: Barrier iteration count
    BARRIER_PRIMOBJ: Barrier iterate primal objective
    BARRIER_DUALOBJ: Barrier iterate dual objective
    BARRIER_PRIMINF: Barrier iterate primal infeasibility
    BARRIER_DUALINF: Barrier iterate dual infeasibility
    BARRIER_COMPL: Barrier iterate complementarity violation
    MIP_OBJBST: Best known objective bound
    MIP_OBJBND: Best known feasible objective
    MIP_NODCNT: Nodes explored so far
    MIP_SOLCNT: Solutions found so far
    MIP_CUTCNT: Cuts added to the model so far
    MIP_NODLFT: Unexplored nodes
    MIP_ITRCNT: Simplex iterations performed so far
    MIP_OPENSCENARIOS: Number of scenarios that are still open in a multi-scenario model
    MIP_PHASE: Solution phase
    MIPSOL_SOL: Feasible solution (a vector)
    MIPSOL_OBJ: Objective value for feasible solution
    MIPSOL_OBJBST: Best known objective bound
    MIPSOL_OBJBND: Best known feasible objective
    MIPSOL_NODCNT: Node count for feasible solution
    MIPSOL_SOLCNT: Solutions found so far
    MIPSOL_OPENSCENARIOS: Number of scenarios that are still open in a multi-scenario model
    MIPSOL_PHASE: Solution phase
    MIPNODE_STATUS: Optimization status of node relaxation
    MIPNODE_REL: Node relaxation solution or ray if unbounded
    MIPNODE_OBJBST: Best known objective bound
    MIPNODE_OBJBND: Best known feasible objective
    MIPNODE_NODCNT: Nodes explored so far
    MIPNODE_SOLCNT: Solutions found so far
    MIPNODE_SBRVAR: Node branching variable
    MIPNODE_OPENSCENARIOS: Number of scenarios that are still open in a multi-scenario model
    MIPNODE_PHASE: Solution phase
    MULTIOBJ_OBJCNT: Objective count optimized so far
    MULTIOBJ_SOLCNT: Solutions found so far
    MULTIOBJ_SOL: Feasible solution (a vector)
    MULTIOBJ_ITRCNT: Iteration count of last single objective solve
    MULTIOBJ_OBJBST: Objective value of best solution of last objective solve
    MULTIOBJ_OBJBND: Dual bound of last objective solve
    MULTIOBJ_STATUS: Status of last objective solve
    MULTIOBJ_MIPGAP: MIP gap of last single objective solve
    MULTIOBJ_NODCNT: node count of last objective solve
    MULTIOBJ_NODLFT: open node count of last objective solve
    MULTIOBJ_RUNTIME: runtime of last single objective solve
    MULTIOBJ_WORK: work of last objective solve
    IIS_CONSTRMIN: Minimum number of constraints in IIS
    IIS_CONSTRMAX: Maximum number of constraints in IIS
    IIS_CONSTRGUESS: Estimated number of constraints in IIS
    IIS_BOUNDMIN: Minimum number of bounds in IIS
    IIS_BOUNDMAX: Maximum number of bounds in IIS
    IIS_BOUNDGUESS: Estimated number of bounds in IIS
    NLBAR_ITRCNT: NL barrier iteration count
    NLBAR_PRIMOBJ: NL barrier iterate primal objective
    NLBAR_PRIMINF: NL barrier iterate primal infeasibility
    NLBAR_DUALINF: NL barrier iterate dual infeasibility
    NLBAR_COMPL: NL barrier iterate complementarity violation
    MSG_STRING: Output message

  Your callback method can call other methods on the model object:
    cbCut(), cbGet(), cbGetNodeRel(), cbGetSolution(), cbSetSolution()
  '''

  def __setattr__(self, name, value):
    raise AttributeError("Gurobi callback constants are not modifiable")

# Callbacks

  POLLING       =     0
  PRESOLVE      =     1
  SIMPLEX       =     2
  MIP           =     3
  MIPSOL        =     4
  MIPNODE       =     5
  MESSAGE       =     6
  BARRIER       =     7
  MULTIOBJ      =     8
  IIS           =     9
  PDHG          =    10
  NLBAR         =    11

  PRE_COLDEL     =  1000
  PRE_ROWDEL     =  1001
  PRE_SENCHG     =  1002
  PRE_BNDCHG     =  1003
  PRE_COECHG     =  1004
  SPX_ITRCNT     =  2000
  SPX_OBJVAL     =  2001
  SPX_PRIMINF    =  2002
  SPX_DUALINF    =  2003
  SPX_ISPERT     =  2004

  BARRIER_ITRCNT  =  7001
  BARRIER_PRIMOBJ =  7002
  BARRIER_DUALOBJ =  7003
  BARRIER_PRIMINF =  7004
  BARRIER_DUALINF =  7005
  BARRIER_COMPL   =  7006

  MIP_OBJBST             =  3000
  MIP_OBJBND             =  3001
  MIP_NODCNT             =  3002
  MIP_SOLCNT             =  3003
  MIP_CUTCNT             =  3004
  MIP_NODLFT             =  3005
  MIP_ITRCNT             =  3006
  MIP_OPENSCENARIOS      =  3007
  MIP_PHASE              =  3008
  MIPSOL_SOL             =  4001
  MIPSOL_OBJ             =  4002
  MIPSOL_OBJBST          =  4003
  MIPSOL_OBJBND          =  4004
  MIPSOL_NODCNT          =  4005
  MIPSOL_SOLCNT          =  4006
  MIPSOL_OPENSCENARIOS   =  4007
  MIPSOL_PHASE           =  4008
  MIPNODE_STATUS         =  5001
  MIPNODE_REL            =  5002
  MIPNODE_OBJBST         =  5003
  MIPNODE_OBJBND         =  5004
  MIPNODE_NODCNT         =  5005
  MIPNODE_SOLCNT         =  5006
  MIPNODE_BRVAR          =  5007
  MIPNODE_OPENSCENARIOS  =  5008
  MIPNODE_PHASE          =  5009
  MSG_STRING             =  6001
  RUNTIME                =  6002
  WORK                   =  6003
  MEMUSED                =  6004
  MAXMEMUSED             =  6005

  MULTIOBJ_OBJCNT =  8001
  MULTIOBJ_SOLCNT =  8002
  MULTIOBJ_SOL    =  8003
  MULTIOBJ_ITRCNT =  8004
  MULTIOBJ_OBJBST =  8005
  MULTIOBJ_OBJBND =  8006
  MULTIOBJ_STATUS =  8007
  MULTIOBJ_MIPGAP =  8008
  MULTIOBJ_NODCNT =  8009
  MULTIOBJ_NODLFT =  8010
  MULTIOBJ_RUNTIME = 8011
  MULTIOBJ_WORK   =  8012

  IIS_CONSTRMIN   = 9001
  IIS_CONSTRMAX   = 9002
  IIS_CONSTRGUESS = 9003
  IIS_BOUNDMIN    = 9004
  IIS_BOUNDMAX    = 9005
  IIS_BOUNDGUESS  = 9006

  PDHG_ITRCNT     = 10001
  PDHG_PRIMOBJ    = 10002
  PDHG_DUALOBJ    = 10003
  PDHG_PRIMINF    = 10004
  PDHG_DUALINF    = 10005
  PDHG_COMPL      = 10006

  NLBAR_ITRCNT  =  11001
  NLBAR_PRIMOBJ =  11002
  NLBAR_PRIMINF =  11003
  NLBAR_DUALINF =  11004
  NLBAR_COMPL   =  11005
