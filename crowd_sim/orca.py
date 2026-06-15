"""Pure-Python ORCA (Optimal Reciprocal Collision Avoidance).

A faithful, dependency-free port of the RVO2 velocity computation
(van den Berg et al. 2011, "Reciprocal n-body collision avoidance") — the same
algorithm the paper's CrowdSim uses via Python-RVO2, but with no C++/Cython
build. Only the agent-agent case is implemented (no static obstacle lines),
which is all the crowd-navigation circle-crossing scenario needs.

Used by ``crowd_sim.crowd_env`` to move pedestrians: each pedestrian runs ORCA
against the OTHER pedestrians only (the robot is invisible to them — the paper's
CrowdNav regime — so it is never added as a neighbour).

Reference: RVO2 ``Agent::computeNewVelocity`` + ``linearProgram1/2/3``.
"""

import math

import numpy as np

_EPS = 1e-5


def _det(a, b):
    return a[0] * b[1] - a[1] * b[0]


def _abs_sq(v):
    return float(v[0] * v[0] + v[1] * v[1])


def _linear_program1(lines, line_no, radius, opt_velocity, direction_opt, result):
    """Optimize the new velocity along the boundary of constraint ``line_no``,
    subject to the max-speed circle and the previous constraints. Returns True
    on success and writes the solution into ``result`` (a length-2 ndarray)."""
    line = lines[line_no]
    dot_product = float(np.dot(line['point'], line['direction']))
    discriminant = dot_product * dot_product + radius * radius - _abs_sq(line['point'])

    if discriminant < 0.0:
        # Max speed circle fully invalidates this line.
        return False

    sqrt_discriminant = math.sqrt(discriminant)
    t_left = -dot_product - sqrt_discriminant
    t_right = -dot_product + sqrt_discriminant

    for i in range(line_no):
        denominator = _det(line['direction'], lines[i]['direction'])
        numerator = _det(lines[i]['direction'], line['point'] - lines[i]['point'])

        if abs(denominator) <= _EPS:
            # Lines line_no and i are (almost) parallel.
            if numerator < 0.0:
                return False
            continue

        t = numerator / denominator
        if denominator >= 0.0:
            t_right = min(t_right, t)
        else:
            t_left = max(t_left, t)

        if t_left > t_right:
            return False

    if direction_opt:
        if float(np.dot(opt_velocity, line['direction'])) > 0.0:
            res = line['point'] + t_right * line['direction']
        else:
            res = line['point'] + t_left * line['direction']
    else:
        t = float(np.dot(line['direction'], opt_velocity - line['point']))
        if t < t_left:
            res = line['point'] + t_left * line['direction']
        elif t > t_right:
            res = line['point'] + t_right * line['direction']
        else:
            res = line['point'] + t * line['direction']

    result[0], result[1] = float(res[0]), float(res[1])
    return True


def _linear_program2(lines, radius, opt_velocity, direction_opt, result):
    """Find the velocity closest to ``opt_velocity`` inside the max-speed circle
    and all half-plane constraints. Returns the index of the first failing line
    (== len(lines) on full success)."""
    if direction_opt:
        result[:] = opt_velocity * radius
    elif _abs_sq(opt_velocity) > radius * radius:
        result[:] = (opt_velocity / math.sqrt(_abs_sq(opt_velocity))) * radius
    else:
        result[:] = opt_velocity

    for i in range(len(lines)):
        if _det(lines[i]['direction'], lines[i]['point'] - result) > 0.0:
            # result does not satisfy constraint i — re-optimize on its boundary.
            temp = result.copy()
            if not _linear_program1(lines, i, radius, opt_velocity, direction_opt, result):
                result[:] = temp
                return i
    return len(lines)


def _linear_program3(lines, num_obst_lines, begin_line, radius, result):
    """3D fallback for the infeasible (over-constrained / dense) case: minimize
    the maximum constraint violation. Mutates ``result`` in place."""
    distance = 0.0
    for i in range(begin_line, len(lines)):
        if _det(lines[i]['direction'], lines[i]['point'] - result) > distance:
            proj_lines = list(lines[:num_obst_lines])
            for j in range(num_obst_lines, i):
                determinant = _det(lines[i]['direction'], lines[j]['direction'])
                if abs(determinant) <= _EPS:
                    if float(np.dot(lines[i]['direction'], lines[j]['direction'])) > 0.0:
                        continue  # Same direction — line j is redundant.
                    point = 0.5 * (lines[i]['point'] + lines[j]['point'])
                else:
                    point = lines[i]['point'] + (
                        _det(lines[j]['direction'], lines[i]['point'] - lines[j]['point'])
                        / determinant
                    ) * lines[i]['direction']
                direction = lines[j]['direction'] - lines[i]['direction']
                direction = direction / math.sqrt(_abs_sq(direction))
                proj_lines.append({'point': point, 'direction': direction})

            temp = result.copy()
            opt = np.array([-lines[i]['direction'][1], lines[i]['direction'][0]])
            if _linear_program2(proj_lines, radius, opt, True, result) < len(proj_lines):
                # Should not normally happen; keep the best-so-far result.
                result[:] = temp
            distance = _det(lines[i]['direction'], lines[i]['point'] - result)


def orca_new_velocity(pos, vel, radius, pref_vel, neighbor_positions, neighbor_velocities, neighbor_radii, max_speed,
                      time_horizon=3.0, time_step=0.25, responsibility=0.5):
    """Compute one agent's ORCA-collision-free velocity.

    Args:
        pos, vel: this agent's position/velocity, length-2 ndarrays.
        radius: this agent's radius.
        pref_vel: preferred velocity (toward goal at preferred speed), length-2.
        neighbor_positions: positions of the OTHER agents.
        neighbor_velocities: velocities of the OTHER agents.
        neighbor_radii: radii of the OTHER agents.
        max_speed: this agent's maximum speed.
        time_horizon: ORCA planning horizon (s).
        time_step: simulation step (s).
        responsibility: share of avoidance this agent takes (0..1). The default
            0.5 is reciprocal ORCA (each pair splits the burden) — correct for
            pedestrians who mutually avoid. The robot expert (il.expert) uses 1.0
            because pedestrians are invisible to it (they never yield), so the
            robot must take the WHOLE avoidance burden or it under-avoids.
    Returns:
        length-2 ndarray: the new collision-free velocity.
    """
    lines = []
    inv_time_horizon = 1.0 / time_horizon

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

    result = np.zeros(2)
    line_fail = _linear_program2(lines, max_speed, pref_vel, False, result)
    if line_fail < len(lines):
        _linear_program3(lines, 0, line_fail, max_speed, result)
    return result


def _orca_velocities_scalar(positions, velocities, radii, pref_velocities, max_speeds,
                            time_horizon=3.0, time_step=0.25):
    """Scalar reference wrapper: compute new ORCA velocities for ALL agents
    at once (each avoids every OTHER agent). Inputs are (N,2)/(N,)/(N,2)/(N,)
    arrays. Returns an (N,2) array of new velocities."""
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    radii = np.asarray(radii, dtype=float)
    pref_velocities = np.asarray(pref_velocities, dtype=float)
    max_speeds = np.asarray(max_speeds, dtype=float)
    n = len(positions)
    out = np.zeros((n, 2))
    idx = np.arange(n)  # precomputed once; masked per-agent below to drop self
    for i in range(n):
        mask = idx != i
        out[i] = orca_new_velocity(
            positions[i], velocities[i], float(radii[i]), pref_velocities[i],
            positions[mask], velocities[mask], radii[mask], float(max_speeds[i]), time_horizon, time_step,
        )
    return out


def _orca_lines_vectorized(positions, velocities, radii, time_horizon, time_step,
                           responsibility=0.5):
    """Build every ordered agent-neighbor ORCA line with NumPy broadcasting."""
    rel_position = positions[np.newaxis, :, :] - positions[:, np.newaxis, :]
    rel_velocity = velocities[:, np.newaxis, :] - velocities[np.newaxis, :, :]
    rel_x = rel_position[:, :, 0]
    rel_y = rel_position[:, :, 1]
    rel_vx = rel_velocity[:, :, 0]
    rel_vy = rel_velocity[:, :, 1]
    dist_sq = rel_x * rel_x + rel_y * rel_y
    combined_radius = radii[:, np.newaxis] + radii[np.newaxis, :]
    combined_radius_sq = combined_radius * combined_radius

    direction = np.empty_like(rel_position)
    u = np.empty_like(rel_position)

    inv_time_horizon = 1.0 / time_horizon
    w_x = rel_vx - inv_time_horizon * rel_x
    w_y = rel_vy - inv_time_horizon * rel_y
    w_length_sq = w_x * w_x + w_y * w_y
    dot1 = w_x * rel_x + w_y * rel_y

    outside = dist_sq > combined_radius_sq
    cutoff = outside & (dot1 < 0.0) & (
        dot1 * dot1 > combined_radius_sq * w_length_sq
    )
    if np.any(cutoff):
        w_length = np.sqrt(w_length_sq[cutoff])
        unit_w_x = w_x[cutoff] / w_length
        unit_w_y = w_y[cutoff] / w_length
        direction[cutoff, 0] = unit_w_y
        direction[cutoff, 1] = -unit_w_x
        u_scale = combined_radius[cutoff] * inv_time_horizon - w_length
        u[cutoff, 0] = u_scale * unit_w_x
        u[cutoff, 1] = u_scale * unit_w_y

    legs = outside & ~cutoff
    if np.any(legs):
        leg = np.sqrt(dist_sq[legs] - combined_radius_sq[legs])
        leg_rel_x = rel_x[legs]
        leg_rel_y = rel_y[legs]
        leg_radius = combined_radius[legs]
        leg_dist_sq = dist_sq[legs]
        leg_w_x = w_x[legs]
        leg_w_y = w_y[legs]
        left = leg_rel_x * leg_w_y - leg_rel_y * leg_w_x > 0.0

        leg_direction_x = np.empty_like(leg)
        leg_direction_y = np.empty_like(leg)
        leg_direction_x[left] = (
            leg_rel_x[left] * leg[left] - leg_rel_y[left] * leg_radius[left]
        ) / leg_dist_sq[left]
        leg_direction_y[left] = (
            leg_rel_x[left] * leg_radius[left] + leg_rel_y[left] * leg[left]
        ) / leg_dist_sq[left]
        right = ~left
        leg_direction_x[right] = -(
            leg_rel_x[right] * leg[right] + leg_rel_y[right] * leg_radius[right]
        ) / leg_dist_sq[right]
        leg_direction_y[right] = -(
            -leg_rel_x[right] * leg_radius[right] + leg_rel_y[right] * leg[right]
        ) / leg_dist_sq[right]

        direction[legs, 0] = leg_direction_x
        direction[legs, 1] = leg_direction_y
        dot2 = rel_vx[legs] * leg_direction_x + rel_vy[legs] * leg_direction_y
        u[legs, 0] = dot2 * leg_direction_x - rel_vx[legs]
        u[legs, 1] = dot2 * leg_direction_y - rel_vy[legs]

    colliding = ~outside
    if np.any(colliding):
        inv_time_step = 1.0 / time_step
        collision_w_x = rel_vx[colliding] - inv_time_step * rel_x[colliding]
        collision_w_y = rel_vy[colliding] - inv_time_step * rel_y[colliding]
        collision_w_length = np.sqrt(
            collision_w_x * collision_w_x + collision_w_y * collision_w_y
        )
        collision_rel_x = rel_x[colliding]
        collision_rel_y = rel_y[colliding]
        collision_dist_sq = dist_sq[colliding]

        unit_w_x = np.empty_like(collision_w_length)
        unit_w_y = np.empty_like(collision_w_length)
        moving = collision_w_length > _EPS
        unit_w_x[moving] = collision_w_x[moving] / collision_w_length[moving]
        unit_w_y[moving] = collision_w_y[moving] / collision_w_length[moving]

        stationary = ~moving
        rel_length = np.sqrt(collision_dist_sq[stationary])
        separated = rel_length > _EPS
        stationary_unit_x = np.ones_like(rel_length)
        stationary_unit_y = np.zeros_like(rel_length)
        stationary_unit_x[separated] = (
            collision_rel_x[stationary][separated] / rel_length[separated]
        )
        stationary_unit_y[separated] = (
            collision_rel_y[stationary][separated] / rel_length[separated]
        )
        unit_w_x[stationary] = stationary_unit_x
        unit_w_y[stationary] = stationary_unit_y
        collision_w_length[stationary] = 0.0

        direction[colliding, 0] = unit_w_y
        direction[colliding, 1] = -unit_w_x
        u_scale = (
            combined_radius[colliding] * inv_time_step - collision_w_length
        )
        u[colliding, 0] = u_scale * unit_w_x
        u[colliding, 1] = u_scale * unit_w_y

    points = velocities[:, np.newaxis, :] + responsibility * u
    return points, direction


def _orca_velocities_vectorized(positions, velocities, radii, pref_velocities, max_speeds,
                                time_horizon=3.0, time_step=0.25):
    """Compute all ordered-pair ORCA constraints in arrays, then solve each LP."""
    positions = np.asarray(positions, dtype=float)
    velocities = np.asarray(velocities, dtype=float)
    radii = np.asarray(radii, dtype=float)
    pref_velocities = np.asarray(pref_velocities, dtype=float)
    max_speeds = np.asarray(max_speeds, dtype=float)
    n = len(positions)
    out = np.zeros((n, 2))
    if n == 0:
        return out

    points, directions = _orca_lines_vectorized(
        positions, velocities, radii, time_horizon, time_step,
    )
    idx = np.arange(n)
    for i in range(n):
        mask = idx != i
        lines = [
            {'point': point, 'direction': line_direction}
            for point, line_direction in zip(points[i, mask], directions[i, mask])
        ]
        line_fail = _linear_program2(
            lines, float(max_speeds[i]), pref_velocities[i], False, out[i],
        )
        if line_fail < len(lines):
            _linear_program3(lines, 0, line_fail, float(max_speeds[i]), out[i])
    return out


orca_velocities = _orca_velocities_vectorized
