import gymnasium as gym
from gymnasium import spaces
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

class CrowdSimEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, num_humans=5, time_step=0.25, max_time=50.0, scenario='circle', human_dodge_robot=False, randomize_layout=True):
        super(CrowdSimEnv, self).__init__()

        self.scenario = scenario  # 'easy', 'medium', 'hard', 'extreme', 'circle', 'random'
        self.human_dodge_robot = human_dodge_robot
        # When True (default), robot start/goal and pedestrian spawns are
        # randomized every reset (circle-crossing with random antipodal points).
        # When False, the legacy fixed (0,-4)->(0,4) scene is reproduced exactly.
        self.randomize_layout = randomize_layout
        self.num_humans = num_humans
        self.time_step = time_step
        self.max_time = max_time
        
        # Robot physical parameters (Turtlebot3 Waffle)
        self.robot_radius = 0.3
        self.robot_vpref = 0.26  # max speed of Waffle (m/s)
        self.robot_wmax = 1.8    # max angular speed (rad/s)
        
        # Human physical parameters
        self.human_radius = 0.3
        self.human_vpref = 0.5  # typical human walking speed (m/s)
        
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
        #   - spatial_edges: [dx_local, dy_local, rel_vx_local, rel_vy_local]
        #     for each human (num_humans, 4) — position AND relative velocity
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
        if self.scenario == 'easy':
            self.human_vpref = 0.15
            scenario_type = 'circle'
        elif self.scenario == 'easy_plus':
            self.human_vpref = 0.20
            scenario_type = 'circle'
        elif self.scenario == 'medium':
            self.human_vpref = 0.30
            scenario_type = 'circle'
        elif self.scenario == 'hard':
            self.human_vpref = 0.50
            scenario_type = 'circle'
        elif self.scenario == 'extreme':
            self.human_vpref = 0.50
            scenario_type = 'random'
        else:
            self.human_vpref = 0.50
            scenario_type = self.scenario
        
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
                    self.humans_gx[i] = -px
                    self.humans_gy[i] = -py
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
        spatial_edges = np.zeros((self.num_humans, 6), dtype=np.float32)
        for i in range(self.num_humans):
            dx_global = self.humans_px[i] - self.robot_px
            dy_global = self.humans_py[i] - self.robot_py
            dvx_global = self.humans_vx[i] - self.robot_vx
            dvy_global = self.humans_vy[i] - self.robot_vy
            # Goal-direction unit vector (global), scale-free intent signal.
            gvx = self.humans_gx[i] - self.humans_px[i]
            gvy = self.humans_gy[i] - self.humans_py[i]
            gnorm = np.hypot(gvx, gvy) + 1e-9
            gdx, gdy = gvx / gnorm, gvy / gnorm
            # Rotate position, relative velocity, and goal direction to local frame
            spatial_edges[i, 0] = dx_global * cos_t + dy_global * sin_t
            spatial_edges[i, 1] = -dx_global * sin_t + dy_global * cos_t
            spatial_edges[i, 2] = dvx_global * cos_t + dvy_global * sin_t
            spatial_edges[i, 3] = -dvx_global * sin_t + dvy_global * cos_t
            spatial_edges[i, 4] = gdx * cos_t + gdy * sin_t
            spatial_edges[i, 5] = -gdx * sin_t + gdy * cos_t
            
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
        
        # Distances between all humans
        d_humans = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                if i != j:
                    d_humans[i, j] = np.hypot(self.humans_px[i] - self.humans_px[j], self.humans_py[i] - self.humans_py[j])
        
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
            w_hj_sum = 0.0
            for j in range(N):
                if j != i:
                    w_hj_sum += 1.0 / (d_humans[i, j] + 1e-5)
                    
            omega = w_hr / (w_hr + w_hj_sum + 1e-5)
            
            # Individual social pressure on human i
            I_2 = omega * I_1

            # Add to total social pressure index. The 1/d_hr factor is capped at
            # 10.0 to prevent runaway penalties when humans cluster very close to
            # the robot (d_hr < 0.1m), which would otherwise dominate the reward.
            inv_d_hr = min(1.0 / (d_hr + 1e-5), 10.0)
            I_sp += inv_d_hr * I_2
            
        return I_sp

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
        # 1. Goal approaching reward. Terminal goal = +50 (a single success is
        # worth ~3× a typical timeout reward, enough to dominate the stall
        # attractor without making collision -20 look "affordable" relative to
        # +100 — the 100.0 setting caused HARD-phase collision rates to spike
        # to 65-95% as agents charged through humans toward the goal).
        # Approach coefficient lowered 10 → 5 so the dense shaping does not
        # drown out the sparse goal signal. Orientation penalty weight 0.3 →
        # 0.05 so that the maximum per-step orientation cost (~0.15) stays
        # well under the maximum per-step approach reward (~0.33 at v_max),
        # preventing the "rotate toward goal but don't move" equilibrium.
        if reached_goal:
            r_g = 20.0
        else:
            prev_dist_to_goal = np.hypot(prev_rx - self.robot_gx, prev_ry - self.robot_gy)
            r_g = 1.0 * (prev_dist_to_goal - dist_to_goal)

            angle_to_goal = np.arctan2(self.robot_gy - self.robot_py, self.robot_gx - self.robot_px)
            angle_diff = angle_to_goal - self.robot_theta
            angle_diff = (angle_diff + np.pi) % (2.0 * np.pi) - np.pi
            weight = 0.05 * np.clip((d_min - 0.6) / 1.4, 0.0, 1.0)
            r_g -= weight * np.abs(angle_diff)

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
        r_s = -6.0 * I_sp
        
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
        Updates pedestrian positions using a robust Social Force Model (SFM).
        """
        N = self.num_humans
        tau = 0.5  # relaxation time
        A = 2.0    # repulsive force magnitude
        B = 0.3    # repulsive force range
        
        new_px = np.zeros(N)
        new_py = np.zeros(N)
        new_vx = np.zeros(N)
        new_vy = np.zeros(N)
        
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
                pref_vx = (dx_g / dist_g) * self.human_vpref
                pref_vy = (dy_g / dist_g) * self.human_vpref
                
            f_drive_x = (pref_vx - vx) / tau
            f_drive_y = (pref_vy - vy) / tau
            
            # 2. Repulsion from other humans
            f_rep_x = 0.0
            f_rep_y = 0.0
            for j in range(N):
                if j != i:
                    dx = px - self.humans_px[j]
                    dy = py - self.humans_py[j]
                    dist = np.hypot(dx, dy)
                    r_sum = 2.0 * self.human_radius
                    # Repulsive force points away from human j
                    if dist > 0:
                        force = A * np.exp((r_sum - dist) / B)
                        f_rep_x += force * (dx / dist)
                        f_rep_y += force * (dy / dist)
                        
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
            if speed > self.human_vpref:
                nvx = (nvx / speed) * self.human_vpref
                nvy = (nvy / speed) * self.human_vpref
                
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
