# Wrapper for 'GRB.Error' object

class ErrorConstClass(object):
  '''
  Gurobi error codes (e.g., exception.errno == GRB.ERROR_OUT_OF_MEMORY):

    OUT_OF_MEMORY: Exhausted available memory
    NULL_ARGUMENT: NULL argument value
    INVALID_ARGUMENT: Invalid argument value
    UNKNOWN_ATTRIBUTE: Unknown attribute name
    DATA_NOT_AVAILABLE: Requested data not available
    INDEX_OUT_OF_RANGE: Constr/var index out of range
    UNKNOWN_PARAMETER: Unknown parameter name
    VALUE_OUT_OF_RANGE: Parameter value outside of valid range
    NO_LICENSE: No Gurobi license found
    SIZE_LIMIT_EXCEEDED: Exceeded licensed model size limit
    CALLBACK: Error in callback
    FILE_READ: Error reading file
    FILE_WRITE: Error writing file
    NUMERIC: Numeric error encountered
    IIS_NOT_INFEASIBLE: Can't compute an IIS on a feasible model
    NOT_FOR_MIP: Method not valid for MIP models
    OPTIMIZATION_IN_PROGRESS: Must stop optimization to query results
    DUPLICATES: Duplicate entries in list
    NODEFILE: Problem reading or writing node file
    Q_NOT_PSD: Q matrix isn't Positive Semi-Definite
    QCP_EQUALITY_CONSTRAINT: Quadratic constraints must be inequalities
    NETWORK: Network error
    JOB_REJECTED: Job rejected by Compute Server
    NOT_SUPPORTED: Requested operation is not supported
    EXCEED_2B_NONZEROS: Result too large for query routine
    INVALID_PIECEWISE_OBJ: Problem with specified piecewise-linear objective
    JOB_REJECTED: Job rejected by Compute Server
    NOT_SUPPORTED: Operation is not supported
    NOT_IN_MODEL: Variable/constraint not in model
    FAILED_TO_CREATE_MODEL: Failed to create the requested model
    CLOUD: Instant Cloud error
    MODEL_MODIFICATION: An error occurred during model modification or update
    CSWORKER: An error occurred with the client-server application
    TUNE_MODEL_TYPES: Tried multi-model tuning on models of different types
    INTERNAL: Internal Gurobi error
  '''

  def __setattr__(self, name, value):
    raise AttributeError("Gurobi error code constants are not modifiable")

  OUT_OF_MEMORY            = 10001
  NULL_ARGUMENT            = 10002
  INVALID_ARGUMENT         = 10003
  UNKNOWN_ATTRIBUTE        = 10004
  DATA_NOT_AVAILABLE       = 10005
  INDEX_OUT_OF_RANGE       = 10006
  UNKNOWN_PARAMETER        = 10007
  VALUE_OUT_OF_RANGE       = 10008
  NO_LICENSE               = 10009
  SIZE_LIMIT_EXCEEDED      = 10010
  CALLBACK                 = 10011
  FILE_READ                = 10012
  FILE_WRITE               = 10013
  NUMERIC                  = 10014
  IIS_NOT_INFEASIBLE       = 10015
  NOT_FOR_MIP              = 10016
  OPTIMIZATION_IN_PROGRESS = 10017
  DUPLICATES               = 10018
  NODEFILE                 = 10019
  Q_NOT_PSD                = 10020
  QCP_EQUALITY_CONSTRAINT  = 10021
  NETWORK                  = 10022
  JOB_REJECTED             = 10023
  NOT_SUPPORTED            = 10024
  EXCEED_2B_NONZEROS       = 10025
  INVALID_PIECEWISE_OBJ    = 10026
  UPDATEMODE_CHANGE        = 10027
  CLOUD                    = 10028
  MODEL_MODIFICATION       = 10029
  CSWORKER                 = 10030
  TUNE_MODEL_TYPES         = 10031
  NOT_IN_MODEL             = 20001
  FAILED_TO_CREATE_MODEL   = 20002
  INTERNAL                 = 20003
