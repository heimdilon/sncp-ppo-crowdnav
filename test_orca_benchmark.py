import numpy as np
import time
from crowd_sim.orca import orca_velocities

def test_benchmark_orca_velocities():
    n = 100
    positions = np.random.rand(n, 2)
    velocities = np.random.rand(n, 2)
    radii = np.ones(n) * 0.3
    pref_velocities = np.random.rand(n, 2)
    max_speeds = np.ones(n) * 1.0

    start_time = time.time()
    for _ in range(100):
        orca_velocities(positions, velocities, radii, pref_velocities, max_speeds)
    end_time = time.time()
    print(f"\nTime taken for 100 calls with n={n}: {end_time - start_time:.4f} seconds")

if __name__ == "__main__":
    test_benchmark_orca_velocities()
