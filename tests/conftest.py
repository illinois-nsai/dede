import random
import numpy as np
import os
import pytest

@pytest.fixture(autouse=True, scope="session")
def set_global_seed():
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    print(f"[pytest] Seed set to {seed}")
