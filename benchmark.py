import timeit
import numpy as np
from crowd_sim.crowd_env import CrowdSimEnv

env = CrowdSimEnv(num_humans=100) # use 100 humans to show a larger effect
env.reset()

setup = """
from __main__ import env
"""

test_code = """
env._get_obs()
"""

# Try getting observation
print("Baseline time:", timeit.timeit(test_code, setup=setup, number=1000))
