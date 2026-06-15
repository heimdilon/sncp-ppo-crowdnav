import numpy as np
import time
from crowd_sim.orca import orca_velocities, orca_new_velocity

def benchmark_velocities(n, iters):
    np.random.seed(42)
    positions = np.random.rand(n, 2)
    velocities = np.random.rand(n, 2)
    radii = np.ones(n) * 0.3
    pref_velocities = np.random.rand(n, 2)
    max_speeds = np.ones(n) * 1.0

    start_time = time.time()
    for _ in range(iters):
        orca_velocities(positions, velocities, radii, pref_velocities, max_speeds)
    end_time = time.time()
    return end_time - start_time

if __name__ == "__main__":
    for n in [10, 30, 50, 100]:
        t = benchmark_velocities(n, 100)
        print(f"N={n}, Iterations=100: Time = {t:.4f} seconds")
