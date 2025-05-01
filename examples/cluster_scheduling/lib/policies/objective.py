from enum import Enum, unique


OBJ_STRS = ["max_min_fairness_perf", "max_proportional_fairness", "gandiva"]


@unique
class Objective(Enum):
    TOTAL_UTIL = 0
    MAX_MIN_ALLOC = 1

    @classmethod
    def get_obj_from_str(cls, obj_str):
        if obj_str == "max_min_fairness_perf":
            return cls.MAX_MIN_ALLOC
        elif obj_str == "max_proportional_fairness":
            return cls.TOTAL_UTIL
        else:
            raise Exception("{} not supported".format(obj_str))
