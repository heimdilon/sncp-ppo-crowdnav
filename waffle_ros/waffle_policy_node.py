import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist, PoseArray
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import torch
import numpy as np
import os

# Import the model from our module
from sncp_ppo.models import SNCPPolicy

class WafflePolicyNode(Node):
    def __init__(self):
        super().__init__('waffle_policy_node')
        
        # Declare parameters
        self.declare_parameter('model_path', 'checkpoints/sncp_ppo.pt')
        self.declare_parameter('num_humans', 5)
        self.declare_parameter('goal_x', 0.0)
        self.declare_parameter('goal_y', 4.0)
        self.declare_parameter('use_lidar_fallback', True)
        
        model_path = self.get_parameter('model_path').get_value()
        self.num_humans = self.get_parameter('num_humans').get_value()
        self.goal_x = self.get_parameter('goal_x').get_value()
        self.goal_y = self.get_parameter('goal_y').get_value()
        self.use_lidar_fallback = self.get_parameter('use_lidar_fallback').get_value()
        
        # Physical parameters
        self.robot_radius = 0.3
        self.robot_vpref = 0.26
        self.robot_wmax = 1.8
        
        # Initialize robot states
        self.robot_px = 0.0
        self.robot_py = 0.0
        self.robot_vx = 0.0
        self.robot_vy = 0.0
        self.robot_theta = 0.0
        
        # Initialize pedestrian states
        self.humans_px = np.zeros(self.num_humans)
        self.humans_py = np.zeros(self.num_humans)
        self.humans_vx = np.zeros(self.num_humans)
        self.humans_vy = np.zeros(self.num_humans)
        
        # Load trained policy model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.get_logger().info(f"Loading policy on device: {self.device}")
        
        self.policy = SNCPPolicy(robot_vpref=self.robot_vpref, robot_wmax=self.robot_wmax).to(self.device)
        if os.path.exists(model_path):
            self.policy.load_state_dict(torch.load(model_path, map_location=self.device))
            self.policy.eval()
            self.get_logger().info(f"Successfully loaded trained policy from {model_path}")
        else:
            self.get_logger().warn(f"Checkpoint not found at {model_path}! Running with unitialized weights.")
            
        # Initialize policy hidden states
        self.h_states = self.policy.init_hidden(batch_size=1, num_humans=self.num_humans, device=self.device)
        
        # ROS 2 Subscriptions and Publications
        self.odom_sub = self.create_subscription(Odometry, '/odom', self.odom_callback, 10)
        self.scan_sub = self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.ped_sub = self.create_subscription(PoseArray, '/pedestrians', self.pedestrians_callback, 10)
        
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        # Timer callback (runs at 4Hz matching 0.25s control step)
        self.control_timer = self.create_timer(0.25, self.control_callback)
        self.get_logger().info("SNCP-PPO Waffle Navigation Node Initialized.")

    def odom_callback(self, msg):
        self.robot_px = msg.pose.pose.position.x
        self.robot_py = msg.pose.pose.position.y
        
        # Calculate yaw from quaternion
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_theta = np.atan2(siny_cosp, cosy_cosp)
        
        # Velocities
        self.robot_vx = msg.twist.twist.linear.x * np.cos(self.robot_theta)
        self.robot_vy = msg.twist.twist.linear.x * np.sin(self.robot_theta)

    def pedestrians_callback(self, msg):
        """
        Receives pedestrian poses from a tracker node.
        """
        if self.use_lidar_fallback:
            # If lidar fallback is enabled, scan_callback will handle pedestrian states
            return
            
        num_received = len(msg.poses)
        for i in range(self.num_humans):
            if i < num_received:
                px = msg.poses[i].position.x
                py = msg.poses[i].position.y
                # Compute simple velocity estimate if tracker doesn't provide it
                dt = 0.25
                self.humans_vx[i] = (px - self.humans_px[i]) / dt
                self.humans_vy[i] = (py - self.humans_py[i]) / dt
                self.humans_px[i] = px
                self.humans_py[i] = py
            else:
                # Fill remaining slots far away
                self.humans_px[i] = 999.0
                self.humans_py[i] = 999.0
                self.humans_vx[i] = 0.0
                self.humans_vy[i] = 0.0

    def scan_callback(self, msg):
        """
        LiDAR scan fallback: clusters laser scan points to detect nearby pedestrians/obstacles.
        """
        if not self.use_lidar_fallback:
            return
            
        ranges = np.array(msg.ranges)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        
        # Filter valid ranges
        valid_indices = np.isfinite(ranges) & (ranges > msg.range_min) & (ranges < 6.0)
        valid_ranges = ranges[valid_indices]
        valid_angles = angles[valid_indices]
        
        # Convert to local Cartesian coordinates
        local_x = valid_ranges * np.cos(valid_angles)
        local_y = valid_ranges * np.sin(valid_angles)
        
        # Transform to global coordinates
        global_x = self.robot_px + local_x * np.cos(self.robot_theta) - local_y * np.sin(self.robot_theta)
        global_y = self.robot_py + local_x * np.sin(self.robot_theta) + local_y * np.cos(self.robot_theta)
        
        points = np.stack([global_x, global_y], axis=1)
        
        # Simple Clustering (Euclidean distance threshold)
        clusters = []
        if len(points) > 0:
            visited = np.zeros(len(points), dtype=bool)
            dist_threshold = 0.5  # meters
            
            for i in range(len(points)):
                if not visited[i]:
                    # Find all points in cluster
                    cluster_points = [points[i]]
                    visited[i] = True
                    
                    # Search neighbors recursively
                    queue = [i]
                    while len(queue) > 0:
                        idx = queue.pop(0)
                        dists = np.linalg.norm(points - points[idx], axis=1)
                        neighbors = np.where((dists < dist_threshold) & (~visited))[0]
                        for n in neighbors:
                            visited[n] = True
                            cluster_points.append(points[n])
                            queue.append(n)
                            
                    clusters.append(np.mean(cluster_points, axis=0))
                    
        # Sort clusters by proximity to robot
        if len(clusters) > 0:
            clusters = np.array(clusters)
            dists_to_robot = np.linalg.norm(clusters - np.array([self.robot_px, self.robot_py]), axis=1)
            sorted_indices = np.argsort(dists_to_robot)
            sorted_clusters = clusters[sorted_indices]
        else:
            sorted_clusters = np.empty((0, 2))
            
        # Update human states
        for i in range(self.num_humans):
            if i < len(sorted_clusters):
                px, py = sorted_clusters[i]
                # Calculate velocity estimate
                dt = 0.25
                self.humans_vx[i] = (px - self.humans_px[i]) / dt
                self.humans_vy[i] = (py - self.humans_py[i]) / dt
                
                # Limit velocity to sensible pedestrian bounds
                speed = np.hypot(self.humans_vx[i], self.humans_vy[i])
                if speed > 1.5:
                    self.humans_vx[i] = (self.humans_vx[i] / speed) * 1.5
                    self.humans_vy[i] = (self.humans_vy[i] / speed) * 1.5
                    
                self.humans_px[i] = px
                self.humans_py[i] = py
            else:
                # Default to far away positions if less than num_humans detected
                self.humans_px[i] = 999.0
                self.humans_py[i] = 999.0
                self.humans_vx[i] = 0.0
                self.humans_vy[i] = 0.0

    def control_callback(self):
        # 1. Check goal condition
        dist_to_goal = np.hypot(self.robot_px - self.goal_x, self.robot_py - self.goal_y)
        if dist_to_goal < self.robot_radius:
            self.get_logger().info("Goal Reached! Stopping robot.")
            self.stop_robot()
            return
            
        # 2. Formulate observation matching gym obs structure
        # Robot node state: [px, py, vx, vy, gx, gy, vpref, radius, theta]
        robot_node = np.array([
            self.robot_px, self.robot_py,
            self.robot_vx, self.robot_vy,
            self.goal_x, self.goal_y,
            self.robot_vpref,
            self.robot_radius,
            self.robot_theta
        ], dtype=np.float32)
        
        # Spatial edges: [dx, dy] relative positions of humans to robot
        spatial_edges = np.zeros((self.num_humans, 2), dtype=np.float32)
        for i in range(self.num_humans):
            spatial_edges[i] = [
                self.humans_px[i] - self.robot_px,
                self.humans_py[i] - self.robot_py
            ]
            
        # Temporal edges: [vx, vy] of robot
        temporal_edges = np.array([self.robot_vx, self.robot_vy], dtype=np.float32)
        
        # Prepare observation tensors
        obs_tensor = {
            'robot_node': torch.tensor(robot_node, dtype=torch.float32, device=self.device).unsqueeze(0),
            'spatial_edges': torch.tensor(spatial_edges, dtype=torch.float32, device=self.device).unsqueeze(0),
            'temporal_edges': torch.tensor(temporal_edges, dtype=torch.float32, device=self.device).unsqueeze(0)
        }
        
        # 3. Evaluate Policy Network
        with torch.no_grad():
            mu, _, _, h_states_next = self.policy(obs_tensor, self.h_states)
            self.h_states = h_states_next
            
        # Extract actions
        action = mu.cpu().numpy()[0]
        v = float(action[0])
        w = float(action[1])
        
        # Apply safety bounds
        v = np.clip(v, 0.0, self.robot_vpref)
        w = np.clip(w, -self.robot_wmax, self.robot_wmax)
        
        # 4. Publish commands
        twist = Twist()
        twist.linear.x = v
        twist.angular.z = w
        self.cmd_pub.publish(twist)
        
        self.get_logger().info(f"Target Goal Dist: {dist_to_goal:.2f}m | Action: v={v:.2f} m/s, w={w:.2f} rad/s")

    def stop_robot(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)

def main(args=None):
    rclpy.init(args=args)
    node = WafflePolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.stop_robot()
        node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
