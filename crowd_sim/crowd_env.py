import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from crowd_sim.orca import orca_velocities

class CrowdSimEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(
        self,
        num_humans=5,
        time_step=0.25,
        max_time=50.0,
        scenario='circle',
        human_dodge_robot=False,
        randomize_layout=True,
        comfort_coeff=6.0,
        human_motion_model='orca',
        robot_vpref=0.26,
        human_vpref_override=None,
        human_goal_noise=0.0,
    ):
        super(CrowdSimEnv, self).__init__()

        self.scenario = scenario  # 'easy', 'medium', 'hard', 'extreme', 'circle', 'random'
        self.human_dodge_robot = human_dodge_robot
        if human_motion_model not in ('sfm', 'linear', 'orca'):
            raise ValueError("human_motion_model must be 'sfm', 'linear', or 'orca'")
        self.human_motion_model = human_motion_model
        # When set, ALL pedestrians use this preferred speed regardless of
        # scenario (used for the paper-reproduction regime: parity at the robot's
        # 1.0 m/s). It overrides the scenario speed in reset() AND lands in the
        # humans_vpref array the motion models actually read, so the speed is
        # genuinely applied (a bare `env.human_vpref = ...` after reset does not
        # update humans_vpref — a latent gotcha this bypasses).
        self.human_vpref_override = human_vpref_override
        # Circle-crossing goals are antipodal (goal = -start), so EVERY path
        # passes through the exact center (0,0) and the crowd funnels into a
        # clump there — unlike the paper, whose human trajectories spread across
        # the central region. human_goal_noise > 0 perturbs each goal by
        # uniform(-noise, noise) per axis so the crossings spread out (paper-like).
        self.human_goal_noise = human_goal_noise
        # When True (default), robot start/goal and pedestrian spawns are
        # randomized every reset (circle-crossing with random antipodal points).
        # When False, the legacy fixed (0,-4)->(0,4) scene is reproduced exactly.
        self.randomize_layout = randomize_layout
        self.num_humans = num_humans
        self.time_step = time_step
        self.max_time = max_time
        self.comfort_coeff = comfort_coeff
        
        # Robot physical parameters (Turtlebot3 Waffle by default; the
        # paper-reproduction run overrides robot_vpref to the paper's 1.0 m/s).
        self.robot_radius = 0.3
        self.robot_vpref = robot_vpref  # max speed (m/s); Waffle hardware = 0.26
        self.robot_wmax = 1.8    # max angular speed (rad/s)
        
        # Human physical parameters
        self.human_radius = 0.3
        self.human_vpref = 0.5  # typical human walking speed (m/s)

        # Social Force Model repulsion constants — single source of truth shared
        # by human-human repulsion (_human_repulsion_forces) and human-robot
        # repulsion (_move_humans), so the two can never silently diverge.
        self.sfm_repulsion_A = 2.0  # repulsive force magnitude
        self.sfm_repulsion_B = 0.3  # repulsive force range
        
        # Safe distance threshold
        self.d_col = 0.3  # collision distance = robot_radius + human_radius
        
        # Ellipse parameters for comfort space
        self.a_front = 1.5
        self.a_back = 0.5
        self.b_left = 0.5
        self.b_right = 1.0  # right side is larger to encourage robot to pass on left (its right)
        
        # Define action space: [v_linear, w_angular]
        self.action_space = spaces.Box(
            low=np.array([0.0, -self.robot_wmax], dtype=np.float32),
            high=np.array([self.robot_vpref, self.robot_wmax], dtype=np.float32),
            dtype=np.float32
        )
        
        # Define observation space as a Dict.
        # ALL observations are in the robot's LOCAL coordinate frame:
        #   - Robot faces along +x axis in local frame
        #   - robot_node: [dg_local_x, dg_local_y, v_linear, dist_to_goal, vpref, radius, w_angular] (7,)
        #   - spatial_edges: [dx_local, dy_local, rel_vx_local, rel_vy_local,
        #     goal_dir_x, goal_dir_y] for each human (num_humans, 6) — position,
        #     relative velocity AND goal-direction intent
        #   - temporal_edges: [v_linear, w_angular] in local frame (2,)
        self.observation_space = spaces.Dict({
            'robot_node': spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32),
            'spatial_edges': spaces.Box(low=-np.inf, high=np.inf, shape=(self.num_humans, 6), dtype=np.float32),
            'temporal_edges': spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32)
        })
        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_time = 0.0
        
        # Robot start/goal. With randomize_layout (default) the robot spawns at
        # a random angle on a circle of radius R with an antipodal goal and a
        # RANDOM heading, so the policy must generalise over geometry (and learn
        # to turn toward the goal) instead of memorising the fixed
        # (0,-4)->(0,4) scene. randomize_layout=False reproduces it exactly.
        R = 4.0
        self.robot_vx = 0.0
        self.robot_vy = 0.0
        if self.randomize_layout:
            theta_r = self.np_random.uniform(0.0, 2.0 * np.pi)
            self.robot_px = R * np.cos(theta_r)
            self.robot_py = R * np.sin(theta_r)
            self.robot_gx = -self.robot_px
            self.robot_gy = -self.robot_py
            self.robot_theta = self.np_random.uniform(-np.pi, np.pi)  # random heading
        else:
            self.robot_px = 0.0
            self.robot_py = -4.0
            self.robot_theta = np.pi / 2.0  # facing North
            self.robot_gx = 0.0
            self.robot_gy = 4.0
        
        # Initialize humans array (fixed size)
        self.humans_px = np.zeros(self.num_humans)
        self.humans_py = np.zeros(self.num_humans)
        self.humans_vx = np.zeros(self.num_humans)
        self.humans_vy = np.zeros(self.num_humans)
        self.humans_theta = np.zeros(self.num_humans)
        self.humans_gx = np.zeros(self.num_humans)
        self.humans_gy = np.zeros(self.num_humans)
        
        # Determine scenario parameters based on difficulty level
        # NB: human_vpref is also overwritten externally by the curriculum loop
        # (train.py); these defaults only apply when reset() is called directly.
        # v15: pedestrian speeds capped at robot parity (<=0.26 m/s) so a slow
        # TurtleBot3 can feasibly avoid a NON-reactive crowd. (v14 used 0.15-0.50;
        # at 0.5 the robot was 2x slower and could not unilaterally dodge.)
        if self.scenario == 'easy':
            self.human_vpref = 0.13
            scenario_type = 'circle'
        elif self.scenario == 'easy_plus':
            self.human_vpref = 0.18
            scenario_type = 'circle'
        elif self.scenario == 'medium':
            self.human_vpref = 0.22
            scenario_type = 'circle'
        elif self.scenario == 'hard':
            self.human_vpref = 0.26
            scenario_type = 'circle'
        elif self.scenario == 'extreme':
            self.human_vpref = 0.26
            scenario_type = 'random'
        else:
            self.human_vpref = 0.26
            scenario_type = self.scenario

        # Paper-reproduction regime: force a flat pedestrian speed (parity with
        # the faster robot) across all scenarios/phases.
        if self.human_vpref_override is not None:
            self.human_vpref = float(self.human_vpref_override)

        self.humans_vpref = np.full(self.num_humans, self.human_vpref, dtype=float)
        
        if scenario_type == 'circle':
            radius = 4.0
            min_safe = self.robot_radius + self.human_radius + 0.5  # 1.1 m
            if self.randomize_layout:
                # Random circle-crossing: each pedestrian gets a random angle on
                # the circle with an antipodal goal, rejection-sampled to stay at
                # least min_safe from BOTH the robot start and goal so no episode
                # begins in collision. This is the variety the fixed i·2π/N
                # placement lacked — a fresh geometry every episode.
                for i in range(self.num_humans):
                    px, py = self.robot_px, self.robot_py
                    for _ in range(100):
                        angle = self.np_random.uniform(0.0, 2.0 * np.pi)
                        px = radius * np.cos(angle)
                        py = radius * np.sin(angle)
                        d_robot = np.hypot(px - self.robot_px, py - self.robot_py)
                        d_goal = np.hypot(px - self.robot_gx, py - self.robot_gy)
                        if d_robot >= min_safe and d_goal >= min_safe:
                            break
                    self.humans_px[i] = px
                    self.humans_py[i] = py
                    gnoise = self.human_goal_noise
                    nx = self.np_random.uniform(-gnoise, gnoise) if gnoise > 0 else 0.0
                    ny = self.np_random.uniform(-gnoise, gnoise) if gnoise > 0 else 0.0
                    self.humans_gx[i] = -px + nx
                    self.humans_gy[i] = -py + ny
                    dx = self.humans_gx[i] - self.humans_px[i]
                    dy = self.humans_gy[i] - self.humans_py[i]
                    self.humans_theta[i] = np.arctan2(dy, dx)
            else:
                # Legacy deterministic placement (angles i·2π/N + tiny jitter),
                # kept for reproducibility of the old fixed scene. The per-spawn
                # safety check shifts an angle by π/N if it would land on the
                # robot start or goal (prevents the historical N=2/N=4 step-1
                # collisions that plagued earlier EASY_PLUS/HARD phases).
                for i in range(self.num_humans):
                    angle = i * 2.0 * np.pi / self.num_humans
                    px = radius * np.cos(angle)
                    py = radius * np.sin(angle)
                    d_robot = np.hypot(px - self.robot_px, py - self.robot_py)
                    d_goal = np.hypot(px - self.robot_gx, py - self.robot_gy)
                    if d_robot < min_safe or d_goal < min_safe:
                        angle = (i + 0.5) * 2.0 * np.pi / self.num_humans
                    angle += self.np_random.uniform(-0.1, 0.1)
                    self.humans_px[i] = radius * np.cos(angle)
                    self.humans_py[i] = radius * np.sin(angle)
                    self.humans_gx[i] = -radius * np.cos(angle)
                    self.humans_gy[i] = -radius * np.sin(angle)
                    dx = self.humans_gx[i] - self.humans_px[i]
                    dy = self.humans_gy[i] - self.humans_py[i]
                    self.humans_theta[i] = np.arctan2(dy, dx)
        else: # random
            for i in range(self.num_humans):
                while True:
                    px = self.np_random.uniform(-5.0, 5.0)
                    py = self.np_random.uniform(-5.0, 5.0)
                    dist_to_robot = np.hypot(px - self.robot_px, py - self.robot_py)
                    if dist_to_robot > 1.5:
                        if i == 0 or np.min(np.hypot(px - self.humans_px[:i], py - self.humans_py[:i])) > 1.0:
                            self.humans_px[i] = px
                            self.humans_py[i] = py
                            break
                self.humans_gx[i] = -px + self.np_random.uniform(-1.0, 1.0)
                self.humans_gy[i] = -py + self.np_random.uniform(-1.0, 1.0)
                dx = self.humans_gx[i] - self.humans_px[i]
                dy = self.humans_gy[i] - self.humans_py[i]
                self.humans_theta[i] = np.arctan2(dy, dx)

        return self._get_obs(), {}

    def _get_obs(self):
        """
        All observations are in the robot's LOCAL coordinate frame.
        Robot faces along the +x axis in local frame. This means:
        - cos(theta), sin(theta) rotation applied to transform global -> local
        - The network receives ego-centric observations, simplifying the
          mapping to actions (v, w) which are also in the robot's frame.
        """
        cos_t = np.cos(self.robot_theta)
        sin_t = np.sin(self.robot_theta)
        
        # Goal vector in global frame
        dg_x_global = self.robot_gx - self.robot_px
        dg_y_global = self.robot_gy - self.robot_py
        
        # Rotate goal vector to local frame
        dg_local_x = dg_x_global * cos_t + dg_y_global * sin_t
        dg_local_y = -dg_x_global * sin_t + dg_y_global * cos_t
        
        dist_to_goal = np.hypot(dg_x_global, dg_y_global)
        
        # Current linear speed magnitude
        v_linear = np.hypot(self.robot_vx, self.robot_vy)
        
        # Current angular velocity (stored from last action)
        w_angular = getattr(self, '_last_w', 0.0)
        
        # robot_node: [dg_local_x, dg_local_y, v_linear, dist_to_goal, vpref, radius, w_angular]
        robot_node = np.array([
            dg_local_x, dg_local_y,
            v_linear,
            dist_to_goal,
            self.robot_vpref,
            self.robot_radius,
            w_angular
        ], dtype=np.float32)
        
        # Spatial edges: per-pedestrian local-frame position, relative velocity
        # (pedestrian - robot), AND goal-direction unit vector. All rotated into
        # the robot's local frame. Layout per row:
        #   [dx_local, dy_local, rel_vx_local, rel_vy_local, goal_dir_x, goal_dir_y]
        # Goal direction gives the policy each pedestrian's INTENT (where it is
        # heading) so it can anticipate trajectories instead of reacting late.
        # Vectorized over all pedestrians at once (was a per-human Python loop).
        dx_global = self.humans_px - self.robot_px
        dy_global = self.humans_py - self.robot_py
        dvx_global = self.humans_vx - self.robot_vx
        dvy_global = self.humans_vy - self.robot_vy
        # Goal-direction unit vector (global), scale-free intent signal.
        gvx = self.humans_gx - self.humans_px
        gvy = self.humans_gy - self.humans_py
        gnorm = np.hypot(gvx, gvy) + 1e-9
        gdx = gvx / gnorm
        gdy = gvy / gnorm
        # Rotate position, relative velocity, and goal direction to local frame
        spatial_edges = np.empty((self.num_humans, 6), dtype=np.float32)
        spatial_edges[:, 0] = dx_global * cos_t + dy_global * sin_t
        spatial_edges[:, 1] = -dx_global * sin_t + dy_global * cos_t
        spatial_edges[:, 2] = dvx_global * cos_t + dvy_global * sin_t
        spatial_edges[:, 3] = -dvx_global * sin_t + dvy_global * cos_t
        spatial_edges[:, 4] = gdx * cos_t + gdy * sin_t
        spatial_edges[:, 5] = -gdx * sin_t + gdy * cos_t
            
        # Temporal edges: robot velocity in local frame = [v_linear, w_angular]
        temporal_edges = np.array([v_linear, w_angular], dtype=np.float32)
        
        return {
            'robot_node': robot_node,
            'spatial_edges': spatial_edges,
            'temporal_edges': temporal_edges
        }

    def _compute_social_pressure(self):
        """
        Computes the Social Pressure Index (I_sp) based on the asymmetric ellipse personal space model.
        """
        I_sp = 0.0
        N = self.num_humans
        
        # Distances between all humans (vectorized NxN). The diagonal is set to
        # inf so the self term contributes 1/(inf+1e-5)=0 to the omega weight sum
        # below, exactly matching the original loop's `j != i` skip. Named *_mat
        # to avoid clashing with the robot-relative dx/dy scalars in the loop.
        dx_mat = self.humans_px[:, np.newaxis] - self.humans_px
        dy_mat = self.humans_py[:, np.newaxis] - self.humans_py
        d_humans = np.hypot(dx_mat, dy_mat)
        np.fill_diagonal(d_humans, np.inf)
        
        for i in range(N):
            # Robot relative to human i
            dx = self.robot_px - self.humans_px[i]
            dy = self.robot_py - self.humans_py[i]
            d_hr = np.hypot(dx, dy)
            
            # Transform to human local frame
            theta_h = self.humans_theta[i]
            x_local = dx * np.cos(theta_h) + dy * np.sin(theta_h)
            y_local = -dx * np.sin(theta_h) + dy * np.cos(theta_h)
            
            phi = np.arctan2(y_local, x_local)
            
            # Select ellipse semi-axes based on quadrant in local frame
            a = self.a_front if x_local >= 0 else self.a_back
            b = self.b_left if y_local >= 0 else self.b_right
            
            # Original interaction space boundary in direction phi
            s_T = (a * b) / (np.sqrt((b * np.cos(phi))**2 + (a * np.sin(phi))**2) + 1e-6)
            
            # Deformed interaction space boundary
            s_prime_T = min(d_hr, s_T)
            
            # Deformation index I_1
            if s_prime_T < s_T:
                ratio = s_prime_T / (s_T + 1e-6)
                I_1 = 1.0 / (1.0 + np.exp(-3.0 * (0.5 - ratio)))
            else:
                I_1 = 0.0
                
            # Compute distance weight omega
            w_hr = 1.0 / (d_hr + 1e-5)
            w_hj_sum = np.sum(1.0 / (d_humans[i] + 1e-5))
                    
            omega = w_hr / (w_hr + w_hj_sum + 1e-5)
            
            # Individual social pressure on human i
            I_2 = omega * I_1

            # Add to total social pressure index. The 1/d_hr factor is capped at
            # 10.0 to prevent runaway penalties when humans cluster very close to
            # the robot (d_hr < 0.1m), which would otherwise dominate the reward.
            inv_d_hr = min(1.0 / (d_hr + 1e-5), 10.0)
            I_sp += inv_d_hr * I_2

        # v19: clamp to the paper's stated range (Sec 3.3: "Isp ranges from 0 to
        # 1"). The summed 1/d_hr term is otherwise unbounded, so a close cluster
        # could push I_sp to ~8 and make the comfort penalty (-comfort_coeff*I_sp)
        # spike to ~-48/step during exploration — drowning the -20 collision
        # signal and over-teaching caution. Bounding it keeps collision the
        # dominant "do not hit" signal while preserving the distance-keeping
        # gradient at the comfort_coeff (unchanged at 6.0) multiplier.
        return float(np.clip(I_sp, 0.0, 1.0))

    def step(self, action):
        # Action is [v, w]
        v = float(action[0])
        w = float(action[1])
        
        # Store angular velocity for local-frame observations
        self._last_w = w
        
        # Update robot orientation and position (differential drive kinematics)
        self.robot_theta += w * self.time_step
        # Normalize theta to [-pi, pi]
        self.robot_theta = (self.robot_theta + np.pi) % (2.0 * np.pi) - np.pi
        
        self.robot_vx = v * np.cos(self.robot_theta)
        self.robot_vy = v * np.sin(self.robot_theta)
        
        # Previous position
        prev_rx = self.robot_px
        prev_ry = self.robot_py
        
        self.robot_px += self.robot_vx * self.time_step
        self.robot_py += self.robot_vy * self.time_step
        
        # Move humans using a simple, robust Social Force Model (SFM)
        self._move_humans()
        
        # Increment time
        self.current_time += self.time_step
        
        # Calculate distances
        distances = np.hypot(self.humans_px - self.robot_px, self.humans_py - self.robot_py)
        d_min = np.min(distances) if len(distances) > 0 else np.inf
        
        # Check terminal conditions
        collision = d_min < (self.robot_radius + self.human_radius)
        
        dist_to_goal = np.hypot(self.robot_px - self.robot_gx, self.robot_py - self.robot_gy)
        reached_goal = dist_to_goal < self.robot_radius
        
        timeout = self.current_time >= self.max_time
        
        # Calculate Reward components
        # 1. Goal reward (paper Eq 18, r_g). Reaching the goal yields +20; every
        # other step yields 2 * (how much closer to the goal we got this step).
        # This is potential-based shaping — it telescopes to 2*(d_0 - d_final),
        # so it never changes the optimal policy regardless of the path taken —
        # and the paper relies on it to solve the sparse-reward failure its
        # authors call out explicitly: the agent "may only learn to avoid humans
        # without making progress toward the target" (Sec 4.2).
        #
        # v15 had drifted away from the paper here in two ways that together
        # caused v16's timeout-dominant failure (N=1 timeout 60% with I_sp≈0.009,
        # i.e. nothing to avoid — pure goal-reaching breakdown):
        #   (a) coefficient halved 2.0 → 1.0 (a misreading of potential shaping:
        #       halving cannot "spare detours", it only uniformly starves the
        #       progress gradient to max +0.065/step at 0.26 m/s); and
        #   (b) an ad-hoc heading penalty `-weight*|angle_diff|` (NOT in the
        #       paper), up to 0.157/step — larger than the entire max progress
        #       reward — which made progress-while-turning net-negative and
        #       taught the policy to optimise heading over arrival.
        # v18 restores the paper's r_g exactly: coefficient 2.0, no heading term.
        if reached_goal:
            r_g = 20.0
        else:
            prev_dist_to_goal = np.hypot(prev_rx - self.robot_gx, prev_ry - self.robot_gy)
            r_g = 2.0 * (prev_dist_to_goal - dist_to_goal)

        # 2. Collision penalty. Raised 20 → 25 to keep deterrence after the
        # goal reward was reduced 100 → 50; ratio collision/goal stays similar
        # so the agent's risk/reward tradeoff is unchanged.
        if collision:
            r_c = -20.0
        else:
            r_c = 0.0
            
        # 3. Comfort penalty (paper Eq 20): r_s = -2 * I_sp. I_sp is the social
        # pressure index summed over humans (per-human cap 10/d_hr), already a
        # normalized 0..1 quantity. The -2 coefficient matches the source paper;
        # the earlier -0.5/N was ~20x weaker at N=5 and let the robot ignore
        # social proximity (it never braked into crowds).
        I_sp = self._compute_social_pressure()
        r_s = -self.comfort_coeff * I_sp
        
        # 4. Standstill penalty removed. Even the softened -0.05 / v<0.03
        # version was sampling-driven (negative Normal samples clip to 0 and
        # trigger it) rather than policy-driven. With approach coefficient
        # now 5.0, a stalled agent simply collects zero approach reward —
        # that's already the right signal.
        r_still = 0.0
        
        reward = r_g + r_c + r_s + r_still
        
        # Done flags
        terminated = collision or reached_goal
        truncated = timeout
        
        # Info dictionary
        info = {
            'success': reached_goal,
            'collision': collision,
            'timeout': timeout and not reached_goal,
            'comfort': r_s,
            'd_min': d_min,
            'I_sp': I_sp
        }
        
        return self._get_obs(), reward, terminated, truncated, info

    def _move_humans(self):
        """
        Updates pedestrian positions. Default is ORCA (v20, paper-faithful: the
        same reciprocal collision avoidance CrowdSim uses); 'sfm' and 'linear'
        are kept for the legacy/custom-map paths.
        """
        if self.human_motion_model == 'orca':
            self._move_humans_orca()
            return
        if self.human_motion_model == 'linear':
            self._move_humans_linear()
            return

        N = self.num_humans
        tau = 0.5  # relaxation time
        A = self.sfm_repulsion_A  # repulsive force magnitude
        B = self.sfm_repulsion_B  # repulsive force range
        
        new_px = np.zeros(N)
        new_py = np.zeros(N)
        new_vx = np.zeros(N)
        new_vy = np.zeros(N)

        # Pairwise human-human repulsion for ALL pedestrians at once (vectorized,
        # replacing the former nested O(N^2) loop). Computed on the current
        # positions, exactly as the per-agent loop read them (self.humans_* is
        # only rewritten after the loop), so each per-i force is unchanged.
        f_rep_x_all, f_rep_y_all = self._human_repulsion_forces()

        for i in range(N):
            px = self.humans_px[i]
            py = self.humans_py[i]
            vx = self.humans_vx[i]
            vy = self.humans_vy[i]
            
            # 1. Goal driving force
            gx = self.humans_gx[i]
            gy = self.humans_gy[i]
            dx_g = gx - px
            dy_g = gy - py
            dist_g = np.hypot(dx_g, dy_g)
            
            if dist_g < 0.1:
                # Reached goal, stop
                pref_vx = 0.0
                pref_vy = 0.0
            else:
                vpref_i = self._human_vpref(i)
                pref_vx = (dx_g / dist_g) * vpref_i
                pref_vy = (dy_g / dist_g) * vpref_i
                
            f_drive_x = (pref_vx - vx) / tau
            f_drive_y = (pref_vy - vy) / tau
            
            # 2. Repulsion from other humans (precomputed vectorized, above)
            f_rep_x = f_rep_x_all[i]
            f_rep_y = f_rep_y_all[i]

            # 3. Repulsion from robot
            if self.human_dodge_robot:
                dx_r = px - self.robot_px
                dy_r = py - self.robot_py
                dist_r = np.hypot(dx_r, dy_r)
                r_sum_r = self.human_radius + self.robot_radius
                if dist_r > 0:
                    force_r = A * np.exp((r_sum_r - dist_r) / B)
                    f_rep_x += force_r * (dx_r / dist_r)
                    f_rep_y += force_r * (dy_r / dist_r)
                
            # Total force
            ax = f_drive_x + f_rep_x
            ay = f_drive_y + f_rep_y
            
            # Update velocity
            nvx = vx + ax * self.time_step
            nvy = vy + ay * self.time_step
            
            # Limit speed to max preferred speed
            speed = np.hypot(nvx, nvy)
            vpref_i = self._human_vpref(i)
            if speed > vpref_i:
                nvx = (nvx / speed) * vpref_i
                nvy = (nvy / speed) * vpref_i
                
            # Update position
            new_px[i] = px + nvx * self.time_step
            new_py[i] = py + nvy * self.time_step
            new_vx[i] = nvx
            new_vy[i] = nvy
            
            # Update orientation
            if np.hypot(nvx, nvy) > 0.01:
                self.humans_theta[i] = np.arctan2(nvy, nvx)
                
        self.humans_px = new_px
        self.humans_py = new_py
        self.humans_vx = new_vx
        self.humans_vy = new_vy

    def _human_repulsion_forces(self):
        """Pairwise Social-Force repulsion between pedestrians, vectorized.

        Replaces the original nested O(N^2) Python loop (one np.hypot + np.exp per
        ordered pair) with a single pass of numpy array ops over the N x N pair
        matrix. The per-element operation order matches the loop exactly
        (``force * (d / dist)``) and the row sum keeps the diagonal (the i == j
        term has zero displacement and contributes exactly 0.0, mirroring the
        loop's ``if j != i`` skip), so the output is numerically equivalent to a
        very tight tolerance (the tests pin atol=1e-12). It is not a guaranteed
        bitwise identity, because numpy's reduction order is not contractual
        across versions/platforms.

        Returns ``(f_rep_x, f_rep_y)``, each a length-N array of the net repulsion
        force on every pedestrian from all the others.
        """
        A = self.sfm_repulsion_A  # repulsive force magnitude
        B = self.sfm_repulsion_B  # repulsive force range
        px = self.humans_px
        py = self.humans_py

        # diff[i, j] = position_i - position_j: the force on i points away from j.
        diff_x = px[:, None] - px[None, :]
        diff_y = py[:, None] - py[None, :]
        dist = np.hypot(diff_x, diff_y)

        # dist == 0 on the diagonal (and for any coincident pair); the original
        # loop skips those via `if dist > 0`. diff is also 0 wherever dist is 0,
        # so the contribution is 0 either way — safe_dist just avoids 0/0 -> nan.
        r_sum = 2.0 * self.human_radius
        safe_dist = np.where(dist > 0.0, dist, 1.0)
        force = A * np.exp((r_sum - dist) / B)
        f_rep_x = (force * (diff_x / safe_dist)).sum(axis=1)
        f_rep_y = (force * (diff_y / safe_dist)).sum(axis=1)
        return f_rep_x, f_rep_y

    def _human_vpref(self, index):
        speeds = getattr(self, 'humans_vpref', None)
        if speeds is None:
            return float(self.human_vpref)
        return float(speeds[index])

    def _move_humans_linear(self):
        """Move custom-map pedestrians with fixed per-agent heading and speed."""
        N = self.num_humans
        new_vx = np.zeros(N)
        new_vy = np.zeros(N)

        for i in range(N):
            speed = self._human_vpref(i)
            theta = self.humans_theta[i]
            new_vx[i] = speed * np.cos(theta)
            new_vy[i] = speed * np.sin(theta)

        self.humans_vx = new_vx
        self.humans_vy = new_vy
        self.humans_px = self.humans_px + new_vx * self.time_step
        self.humans_py = self.humans_py + new_vy * self.time_step

    def _move_humans_orca(self):
        """Move pedestrians with ORCA (v20): each avoids the OTHER pedestrians
        reciprocally, heading to its own goal at its preferred speed.

        The ROBOT is deliberately NOT a neighbour — pedestrians are invisible to
        it ("invisible robot", the paper's CrowdNav regime), so the robot must
        still do all of its own avoidance. ORCA only stops the crowd from
        collapsing into an impassable knot at the antipodal-crossing center,
        which is what the Social Force Model did and what blocked high density.
        """
        N = self.num_humans
        pos = np.stack([self.humans_px, self.humans_py], axis=1)
        vel = np.stack([self.humans_vx, self.humans_vy], axis=1)
        radii = np.full(N, self.human_radius)

        speeds = getattr(self, 'humans_vpref', None)
        if speeds is not None:
            # asarray (not array(copy=False)) so a list/non-float humans_vpref is
            # converted instead of raising under NumPy 2.x, matching the old
            # per-element float(_human_vpref(i)) behaviour.
            max_speeds = np.asarray(speeds, dtype=float)
        else:
            max_speeds = np.full(N, self.human_vpref, dtype=float)

        pref = np.zeros((N, 2))
        for i in range(N):
            dx = self.humans_gx[i] - self.humans_px[i]
            dy = self.humans_gy[i] - self.humans_py[i]
            dist = np.hypot(dx, dy)
            if dist >= 0.1:  # else stay at goal (pref velocity 0)
                vpref_i = max_speeds[i]
                pref[i, 0] = (dx / dist) * vpref_i
                pref[i, 1] = (dy / dist) * vpref_i

        new_vel = orca_velocities(
            pos, vel, radii, pref, max_speeds,
            time_horizon=3.0, time_step=self.time_step,
        )
        self.humans_vx = new_vel[:, 0]
        self.humans_vy = new_vel[:, 1]
        self.humans_px = self.humans_px + self.humans_vx * self.time_step
        self.humans_py = self.humans_py + self.humans_vy * self.time_step

        speed = np.hypot(self.humans_vx, self.humans_vy)
        moving = speed > 0.01
        self.humans_theta = np.where(
            moving, np.arctan2(self.humans_vy, self.humans_vx), self.humans_theta
        )

    def render(self, mode='human'):
        # For simple visual verification
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_xlim(-8, 8)
        ax.set_ylim(-8, 8)
        
        # Draw target
        ax.plot(self.robot_gx, self.robot_gy, 'r*', markersize=12, label='Goal')
        
        # Draw robot
        robot_circle = patches.Circle((self.robot_px, self.robot_py), self.robot_radius, color='y', fill=True, label='Robot')
        ax.add_patch(robot_circle)
        
        # Draw robot heading
        arrow_len = 0.5
        ax.arrow(self.robot_px, self.robot_py, 
                 arrow_len * np.cos(self.robot_theta), 
                 arrow_len * np.sin(self.robot_theta), 
                 head_width=0.1, head_length=0.1, fc='black', ec='black')
        
        # Draw humans
        for i in range(self.num_humans):
            # Draw ellipse comfort space
            # Rotate by human orientation
            deg = np.degrees(self.humans_theta[i])
            ellipse = patches.Ellipse(
                (self.humans_px[i], self.humans_py[i]), 
                width=self.a_front + self.a_back, 
                height=self.b_left + self.b_right, 
                angle=deg, 
                color='blue', alpha=0.1, fill=True
            )
            ax.add_patch(ellipse)
            
            human_circle = patches.Circle((self.humans_px[i], self.humans_py[i]), self.human_radius, color='b', fill=True)
            ax.add_patch(human_circle)
            
            # Draw human heading
            ax.arrow(self.humans_px[i], self.humans_py[i], 
                     arrow_len * np.cos(self.humans_theta[i]), 
                     arrow_len * np.sin(self.humans_theta[i]), 
                     head_width=0.1, head_length=0.1, fc='blue', ec='blue')
            
        plt.grid(True)
        plt.title(f"Time: {self.current_time:.2f}s")
        plt.show()
