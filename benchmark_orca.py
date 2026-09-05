import numpy as np
import time
from crowd_sim.orca import orca_new_velocity, _EPS
import math

def orig_orca_new_velocity(pos, vel, radius, pref_vel, neighbor_positions, neighbor_velocities, neighbor_radii, max_speed,
                           time_horizon=3.0, time_step=0.25, responsibility=0.5):
    lines = []
    inv_time_horizon = 1.0 / time_horizon

    def _abs_sq(v):
        return float(v[0] * v[0] + v[1] * v[1])

    def _det(a, b):
        return a[0] * b[1] - a[1] * b[0]

    for n_pos, n_vel, n_radius in zip(neighbor_positions, neighbor_velocities, neighbor_radii):
        rel_position = np.asarray(n_pos, dtype=float) - pos
        rel_velocity = vel - np.asarray(n_vel, dtype=float)
        dist_sq = _abs_sq(rel_position)
        combined_radius = radius + n_radius
        combined_radius_sq = combined_radius * combined_radius

        if dist_sq > combined_radius_sq:
            # No collision yet — vector from cutoff-circle center to rel velocity.
            w = rel_velocity - inv_time_horizon * rel_position
            w_length_sq = _abs_sq(w)
            dot1 = float(np.dot(w, rel_position))

            if dot1 < 0.0 and dot1 * dot1 > combined_radius_sq * w_length_sq:
                # Project on cutoff circle.
                w_length = math.sqrt(w_length_sq)
                unit_w = w / w_length
                direction = np.array([unit_w[1], -unit_w[0]])
                u = (combined_radius * inv_time_horizon - w_length) * unit_w
            else:
                # Project on legs.
                leg = math.sqrt(dist_sq - combined_radius_sq)
                if _det(rel_position, w) > 0.0:
                    direction = np.array([
                        rel_position[0] * leg - rel_position[1] * combined_radius,
                        rel_position[0] * combined_radius + rel_position[1] * leg,
                    ]) / dist_sq
                else:
                    direction = -np.array([
                        rel_position[0] * leg + rel_position[1] * combined_radius,
                        -rel_position[0] * combined_radius + rel_position[1] * leg,
                    ]) / dist_sq
                dot2 = float(np.dot(rel_velocity, direction))
                u = dot2 * direction - rel_velocity
        else:
            # Already colliding — project on cutoff circle defined by time step.
            inv_time_step = 1.0 / time_step
            w = rel_velocity - inv_time_step * rel_position
            w_length = math.sqrt(_abs_sq(w))
            if w_length > _EPS:
                unit_w = w / w_length
            else:
                # Degenerate: (near-)coincident agents with no relative motion.
                # Pick an arbitrary push-apart direction so the result is finite.
                rp_length = math.sqrt(dist_sq)
                unit_w = (rel_position / rp_length) if rp_length > _EPS else np.array([1.0, 0.0])
                w_length = 0.0
            direction = np.array([unit_w[1], -unit_w[0]])
            u = (combined_radius * inv_time_step - w_length) * unit_w

        # Reciprocal ORCA splits the burden (responsibility=0.5); the robot
        # expert takes the whole burden (1.0) because pedestrians don't yield.
        point = vel + responsibility * u
        lines.append({'point': point, 'direction': direction})

    from crowd_sim.orca import _linear_program2, _linear_program3
    result = np.zeros(2)
    line_fail = _linear_program2(lines, max_speed, pref_vel, False, result)
    if line_fail < len(lines):
        _linear_program3(lines, 0, line_fail, max_speed, result)
    return result

def run_benchmark():
    np.random.seed(42)
    num_neighbors = 20
    pos = np.array([0.0, 0.0])
    vel = np.array([1.0, 1.0])
    radius = 0.3
    pref_vel = np.array([1.0, 1.0])
    neighbor_positions = np.random.rand(num_neighbors, 2) * 5
    neighbor_velocities = np.random.rand(num_neighbors, 2)
    neighbor_radii = np.ones(num_neighbors) * 0.3
    max_speed = 1.0

    # Verify correctness
    orig_res = orig_orca_new_velocity(pos, vel, radius, pref_vel, neighbor_positions, neighbor_velocities, neighbor_radii, max_speed)
    vec_res = orca_new_velocity(pos, vel, radius, pref_vel, neighbor_positions, neighbor_velocities, neighbor_radii, max_speed)

    print("Orig res:", orig_res)
    print("Vec res:", vec_res)
    print("Correctness check:", np.allclose(orig_res, vec_res))

    N = 10000

    start = time.perf_counter()
    for _ in range(N):
        orig_orca_new_velocity(pos, vel, radius, pref_vel, neighbor_positions, neighbor_velocities, neighbor_radii, max_speed)
    end = time.perf_counter()
    orig_time = end - start

    start = time.perf_counter()
    for _ in range(N):
        orca_new_velocity(pos, vel, radius, pref_vel, neighbor_positions, neighbor_velocities, neighbor_radii, max_speed)
    end = time.perf_counter()
    vec_time = end - start

    print(f"Original time: {orig_time:.5f} s")
    print(f"Vectorized time: {vec_time:.5f} s")
    print(f"Speedup: {orig_time / vec_time:.2f}x")

if __name__ == "__main__":
    run_benchmark()
