"""Micro-benchmark for CrowdSimEnv._get_obs (the vectorized spatial_edges path).
Run directly: ``python benchmark.py``."""

import os as _os, sys as _sys  # repo-root path bootstrap (run standalone: python scripts/X.py)
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
import timeit

from crowd_sim.crowd_env import CrowdSimEnv


def main():
    env = CrowdSimEnv(num_humans=100)  # 100 humans to show a larger effect
    env.reset()
    t = timeit.timeit(env._get_obs, number=1000)
    print(f"_get_obs x1000 (N=100): {t:.4f}s")


if __name__ == "__main__":
    main()
