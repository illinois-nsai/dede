# Wrapper for 'GRB.Status' object

class StatusConstClass(object):
  '''
  Gurobi optimization status codes (e.g., model.status == GRB.OPTIMAL):

    LOADED: Model loaded, but no solution information available
    OPTIMAL: Solve to optimality (subject to tolerances)
    INFEASIBLE: Model is infeasible
    INF_OR_UNBD: Model is either infeasible or unbounded
    UNBOUNDED: Model is unbounded
    CUTOFF: Objective is worse than specified cutoff value
    ITERATION_LIMIT: Optimization terminated due to iteration limit
    NODE_LIMIT: Optimization terminated due to node limit
    TIME_LIMIT: Optimization terminated due to time limit
    SOLUTION_LIMIT: Optimization terminated due to solution limit
    INTERRUPTED: User interrupted optimization
    NUMERIC: Optimization terminated due to numerical issues
    SUBOPTIMAL: Optimization terminated with a sub-optimal solution
    INPROGRESS: Optimization currently in progress
    USER_OBJ_LIMIT: Achieved user objective limit
    WORK_LIMIT: Optimization terminated due to work limit
    MEM_LIMIT: Optimization terminated due to soft memory limit
  '''

  def __setattr__(self, name, value):
    raise AttributeError("Gurobi status constants are not modifiable")

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
