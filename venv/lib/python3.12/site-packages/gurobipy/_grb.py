# GRB class - constants

from gurobipy._attrconst import AttrConstClass
from gurobipy._callbackconst import CallbackConstClass
from gurobipy._errorconst import ErrorConstClass
from gurobipy._paramconst import ParamConstClass
from gurobipy._statusconst import StatusConstClass


class GRB(object):
  '''
  Gurobi constants.  Constants defined in this class are as follows:

    Basis status (e.g., var.vBasis == GRB.BASIC):

      BASIC: Variable is basic
      NONBASIC_LOWER: Variable is non-basic at lower bound
      NONBASIC_UPPER: Variable is non-basic at upper bound
      SUPERBASIC: Variable is superbasic

    Constraint sense (e.g., constr.sense = GRB.LESS_EQUAL):

      LESS_EQUAL, GREATER_EQUAL, EQUAL

    Variable types (e.g., var.vType = GRB.INTEGER):

      CONTINUOUS, BINARY, INTEGER, SEMICONT, SEMIINT

    SOS types:

      SOS_TYPE1, SOS_TYPE2

    General constraint types:

      GENCONSTR_MAX, GENCONSTR_MIN, GENCONSTR_ABS, GENCONSTR_AND, GENCONSTR_OR,
      GENCONSTR_NORM, GENCONSTR_NL, GENCONSTR_INDICATOR, GENCONSTR_PWL,
      GENCONSTR_POLY, GENCONSTR_EXP, GENCONSTR_EXPA, GENCONSTR_LOG,
      GENCONSTR_LOGA, GENCONSTR_POW, GENCONSTR_SIN, GENCONSTR_COS, GENCONSTR_TAN

  The GRB class also includes definitions for attributes (GRB.attr),
  errors (GRB.error), parameters (GRB.param), status codes (GRB.status),
  and callbacks (GRB.callback).  You can ask for help on any of these
  (e.g., "help(GRB.attr)").
  '''

  attr     = Attr     = AttrConstClass()
  param    = Param    = ParamConstClass()
  callback = Callback = CallbackConstClass()
  error    = Error    = ErrorConstClass()
  status   = Status   = StatusConstClass()

# Status codes

  LOADED             = 1
  OPTIMAL            = 2
  INFEASIBLE         = 3
  INF_OR_UNBD        = 4
  UNBOUNDED          = 5
  CUTOFF             = 6
  ITERATION_LIMIT    = 7
  NODE_LIMIT         = 8
  TIME_LIMIT         = 9
  SOLUTION_LIMIT     = 10
  INTERRUPTED        = 11
  NUMERIC            = 12
  SUBOPTIMAL         = 13
  INPROGRESS         = 14
  USER_OBJ_LIMIT     = 15
  WORK_LIMIT         = 16
  MEM_LIMIT          = 17
  LOCALLY_OPTIMAL    = 18
  LOCALLY_INFEASIBLE = 19

# Batch status codes

  BATCH_CREATED   = 1
  BATCH_SUBMITTED = 2
  BATCH_ABORTED   = 3
  BATCH_FAILED    = 4
  BATCH_COMPLETED = 5

# Constraint senses

  LESS_EQUAL    = '<'
  GREATER_EQUAL = '>'
  EQUAL         = '='

# Variable types

  CONTINUOUS = 'C'
  BINARY     = 'B'
  INTEGER    = 'I'
  SEMICONT   = 'S'
  SEMIINT    = 'N'

# Objective sense

  MINIMIZE = 1
  MAXIMIZE = -1

# SOS types

  SOS_TYPE1 = 1
  SOS_TYPE2 = 2

# General constraint types

  GENCONSTR_MAX       = 0
  GENCONSTR_MIN       = 1
  GENCONSTR_ABS       = 2
  GENCONSTR_AND       = 3
  GENCONSTR_OR        = 4
  GENCONSTR_NORM      = 5
  GENCONSTR_NL        = 6
  GENCONSTR_INDICATOR = 7
  GENCONSTR_PWL       = 8
  GENCONSTR_POLY      = 9
  GENCONSTR_EXP       = 10
  GENCONSTR_EXPA      = 11
  GENCONSTR_LOG       = 12
  GENCONSTR_LOGA      = 13
  GENCONSTR_POW       = 14
  GENCONSTR_SIN       = 15
  GENCONSTR_COS       = 16
  GENCONSTR_TAN       = 17
  GENCONSTR_LOGISTIC  = 18

# Operation codes
  OPCODE_CONSTANT     = 0
  OPCODE_VARIABLE     = 1
  OPCODE_PLUS         = 2
  OPCODE_MINUS        = 3
  OPCODE_MULTIPLY     = 4
  OPCODE_DIVIDE       = 5
  OPCODE_UMINUS       = 6
  OPCODE_SQUARE       = 7
  OPCODE_SQRT         = 8
  OPCODE_SIN          = 9
  OPCODE_COS          = 10
  OPCODE_TAN          = 11
  OPCODE_POW          = 12
  OPCODE_EXP          = 13
  OPCODE_LOG          = 14
  OPCODE_LOG2         = 15
  OPCODE_LOG10        = 16
  OPCODE_LOGISTIC     = 17
  OPCODE_TANH         = 18
  OPCODE_SIGNPOW      = 19

# Basis status

  BASIC          = 0
  NONBASIC_LOWER = -1
  NONBASIC_UPPER = -2
  SUPERBASIC     = -3

# Numeric constants

  INFINITY  = 1e100
  UNDEFINED = 1e101
  MAXINT    = 2000000000

# Limits

  MAX_NAMELEN    = 255
  MAX_STRLEN     = 512
  MAX_TAGLEN     = 10240
  MAX_CONCURRENT = 64

# Other constants

  DEFAULT_CS_PORT = 61000

# Version number

  VERSION_MAJOR     = 13
  VERSION_MINOR     = 0
  VERSION_TECHNICAL = 1

# Errors

  ERROR_OUT_OF_MEMORY            = 10001
  ERROR_NULL_ARGUMENT            = 10002
  ERROR_INVALID_ARGUMENT         = 10003
  ERROR_UNKNOWN_ATTRIBUTE        = 10004
  ERROR_DATA_NOT_AVAILABLE       = 10005
  ERROR_INDEX_OUT_OF_RANGE       = 10006
  ERROR_UNKNOWN_PARAMETER        = 10007
  ERROR_VALUE_OUT_OF_RANGE       = 10008
  ERROR_NO_LICENSE               = 10009
  ERROR_SIZE_LIMIT_EXCEEDED      = 10010
  ERROR_CALLBACK                 = 10011
  ERROR_FILE_READ                = 10012
  ERROR_FILE_WRITE               = 10013
  ERROR_NUMERIC                  = 10014
  ERROR_IIS_NOT_INFEASIBLE       = 10015
  ERROR_NOT_FOR_MIP              = 10016
  ERROR_OPTIMIZATION_IN_PROGRESS = 10017
  ERROR_DUPLICATES               = 10018
  ERROR_NODEFILE                 = 10019
  ERROR_Q_NOT_PSD                = 10020
  ERROR_QCP_EQUALITY_CONSTRAINT  = 10021
  ERROR_NETWORK                  = 10022
  ERROR_JOB_REJECTED             = 10023
  ERROR_NOT_SUPPORTED            = 10024
  ERROR_EXCEED_2B_NONZEROS       = 10025
  ERROR_INVALID_PIECEWISE_OBJ    = 10026
  ERROR_UPDATEMODE_CHANGE        = 10027
  ERROR_CLOUD                    = 10028
  ERROR_MODEL_MODIFICATION       = 10029
  ERROR_CSWORKER                 = 10030
  ERROR_TUNE_MODEL_TYPES         = 10031
  ERROR_SECURITY                 = 10032
  ERROR_NOT_IN_MODEL             = 20001
  ERROR_FAILED_TO_CREATE_MODEL   = 20002
  ERROR_INTERNAL                 = 20003

# Cuts parameter values

  CUTS_AUTO           = -1
  CUTS_OFF            = 0
  CUTS_CONSERVATIVE   = 1
  CUTS_AGGRESSIVE     = 2
  CUTS_VERYAGGRESSIVE = 3

# Presolve parameter values

  PRESOLVE_AUTO         = -1
  PRESOLVE_OFF          = 0
  PRESOLVE_CONSERVATIVE = 1
  PRESOLVE_AGGRESSIVE   = 2

# Method parameter values

  METHOD_NONE                             = -1
  METHOD_AUTO                             = -1
  METHOD_PRIMAL                           = 0
  METHOD_DUAL                             = 1
  METHOD_BARRIER                          = 2
  METHOD_CONCURRENT                       = 3
  METHOD_DETERMINISTIC_CONCURRENT         = 4
  METHOD_DETERMINISTIC_CONCURRENT_SIMPLEX = 5
  METHOD_PDHG                             = 6

# BarHomogeneous parameter values

  BARHOMOGENEOUS_AUTO = -1
  BARHOMOGENEOUS_OFF  = 0
  BARHOMOGENEOUS_ON   = 1

# BarOrder parameter values

  BARORDER_AUTOMATIC        = -1
  BARORDER_AMD              = 0
  BARORDER_NESTEDDISSECTION = 1

# MIPFocus parameter values

  MIPFOCUS_BALANCED    = 0
  MIPFOCUS_FEASIBILITY = 1
  MIPFOCUS_OPTIMALITY  = 2
  MIPFOCUS_BESTBOUND   = 3

# SimplexPricing parameter values

  SIMPLEXPRICING_AUTO           = -1
  SIMPLEXPRICING_PARTIAL        = 0
  SIMPLEXPRICING_STEEPEST_EDGE  = 1
  SIMPLEXPRICING_DEVEX          = 2
  SIMPLEXPRICING_STEEPEST_QUICK = 3

# VarBranch parameter values

  VARBRANCH_AUTO           = -1
  VARBRANCH_PSEUDO_REDUCED = 0
  VARBRANCH_PSEUDO_SHADOW  = 1
  VARBRANCH_MAX_INFEAS     = 2
  VARBRANCH_STRONG         = 3

# Partition parameter values

  PARTITION_EARLY     = 16
  PARTITION_ROOTSTART = 8
  PARTITION_ROOTEND   = 4
  PARTITION_NODES     = 2
  PARTITION_CLEANUP   = 1

# Callback phase values

  PHASE_MIP_NOREL   = 0
  PHASE_MIP_SEARCH  = 1
  PHASE_MIP_IMPROVE = 2

# FeasRelax method parameter values

  FEASRELAX_LINEAR      = 0
  FEASRELAX_QUADRATIC   = 1
  FEASRELAX_CARDINALITY = 2
