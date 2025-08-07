import random
import numpy as np
import os
import pytest


#SEED = 67 
GUROBI_OPTS = {
    "Threads": 1
}


def set_global_seed():
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)


'''
@pytest.fixture(autouse=True, scope="session")
def set_global_seed_fixed():
    set_global_seed()
'''