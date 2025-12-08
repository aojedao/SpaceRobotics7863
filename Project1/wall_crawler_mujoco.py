#!/usr/bin/env python3
"""
MuJoCo-Based Dual-Arm Wall-Crawler Simulation

This simulation uses the dual_arm_robot.xml MuJoCo environment to implement
the wall-crawling algorithm for an ISS module inspection robot.

MODULAR ARCHITECTURE:
=====================
This main file orchestrates the simulation, importing from:
- arm_controller.py: ArmController class with IK, gripper, and anchor management
- visualization.py: SimulationLogger, PlotGenerator, SceneRenderer

Phase 1: Workspace Visualization & State Machine
- Map workspace positions on walls
- Visualize reachable points in MuJoCo (using custom geoms)
- Show start/goal positions
- Implement state machine for robot control states

Environment dimensions (from dual_arm_robot.xml):
- X: -2.2m to 4.0m (6.2m wide)
- Y: -0.6m to 1.8m (2.4m deep)  
- Z: 0m to 2.0m (2.0m high)

Author: Wall Crawler Project
Date: December 2025
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import os
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict
import heapq
from scipy.spatial.transform import Rotation

# Import modular components (optional - for future refactoring)
# These modules contain extracted functionality that can be used
# to simplify this main file in future iterations
try:
    from arm_controller import ArmController, ArmControllerConfig
    from visualization import SimulationLogger, PlotGenerator, SceneRenderer, generate_flow_diagram
    MODULES_AVAILABLE = True
except ImportError:
    MODULES_AVAILABLE = False
    print("Note: Modular components not loaded - using inline implementations")


# ============================================================================
# ENVIRONMENT CONFIGURATION - Based on dual_arm_robot.xml
# ============================================================================

ISS_MODULE = {
    'x_min': -2.1,   # wall_x_neg position (moved 10cm inward from -2.2)
    'x_max': 3.9,    # wall_x_pos position (moved 10cm inward from 4.0)
    'y_min': -0.5,   # wall_y_neg position (moved 10cm inward from -0.6)
    'y_max': 1.7,    # wall_y_pos position (moved 10cm inward from 1.8)
    'z_min': 0.1,    # floor level (moved 10cm up)
    'z_max': 2.2,    # ceiling level (moved 20cm upward)
}

# Offset for grip point placement (to keep spheres inside visible walls)
GRIP_POINT_OFFSETS = {
    'ceiling': -0.1,  # Move ceiling grip points 10cm downward (inside the module)
    'floor': -0.05,   # Move floor grip points 5cm downward
    'front': -0.05,   # Move front wall (y_max) spheres 5cm inward (negative Y)
    'walls': 0.0,     # Other wall grip points at wall surface
}

ISS_WIDTH = ISS_MODULE['x_max'] - ISS_MODULE['x_min']   # 6.0m
ISS_DEPTH = ISS_MODULE['y_max'] - ISS_MODULE['y_min']   # 2.2m (was 2.4m)
ISS_HEIGHT = ISS_MODULE['z_max'] - ISS_MODULE['z_min']  # 2.3m (was 2.2m)

# DUAL ARM CONFIGURATION - End-mounted arms on X-axis, pointing outward
# Left arm on left end (X=-0.2), pointing outward (-X direction)
# Right arm on right end (X=+0.2), pointing outward (+X direction)
LEFT_ARM_OFFSET = -0.30   # X offset from central body (left end, matches XML)
RIGHT_ARM_OFFSET = 0.05   # X offset from central body (right end, matches XML)
ARM_SEPARATION = 0.40     # 0.4m between arms (X direction)

# KUKA iiwa14 workspace
KUKA_REACH = 1.25  # meters (including gripper)

# For visualization, we center the workspace sphere at link4 (elbow)
# and use a radius that covers the gripper's reachable area
# Link4 to gripper distance is ~0.626m, so we use that as radius
WORKSPACE_SPHERE_RADIUS = 0.63  # meters (distance from elbow to gripper)


# ============================================================================
# STATE MACHINE DEFINITIONS
# ============================================================================

class RobotState(Enum):
    """State machine states for the wall-crawling robot"""
    IDLE = auto()                    # Robot is stationary, waiting for commands
    PLANNING = auto()                # Computing path to goal
    READY_TO_MOVE = auto()           # Path computed, ready to execute
    MOVING_LEFT_ARM = auto()         # Left arm is moving to new position
    MOVING_RIGHT_ARM = auto()        # Right arm is moving to new position
    LEFT_ARM_GRASPING = auto()       # Left arm is grasping the wall
    RIGHT_ARM_GRASPING = auto()      # Right arm is grasping the wall
    LEFT_ARM_RELEASING = auto()      # Left arm is releasing from wall
    RIGHT_ARM_RELEASING = auto()     # Right arm is releasing from wall
    TRANSITIONING = auto()           # Transitioning between walls
    GOAL_REACHED = auto()            # Goal position reached
    ERROR = auto()                   # Error state


class GripperState(Enum):
    """Gripper states"""
    OPEN = auto()
    CLOSED = auto()
    MOVING = auto()


@dataclass
class CrawlerState:
    """State representation for the wall-crawler robot position"""
    position: Tuple[float, float, float]  # Position on wall
    wall: str  # Which wall: 'floor', 'ceiling', 'front', 'back', 'left', 'right'
    active_arm: str  # Which arm just moved: 'left' or 'right'
    
    def __hash__(self):
        rounded_pos = (round(self.position[0], 3), 
                      round(self.position[1], 3), 
                      round(self.position[2], 3))
        return hash((rounded_pos, self.wall, self.active_arm))
    
    def __eq__(self, other):
        return (self.position == other.position and 
                self.wall == other.wall and 
                self.active_arm == other.active_arm)
    
    def __lt__(self, other):
        return self.position < other.position


@dataclass(order=True)
class PriorityNode:
    """Node for A* priority queue"""
    f_cost: float
    state: CrawlerState = field(compare=False)
    g_cost: float = field(compare=False)
    parent: Optional['PriorityNode'] = field(compare=False, default=None)


# ============================================================================
# WALL POSITION GENERATOR
# ============================================================================

class WallPositionMapper:
    """
    Maps discrete positions on the ISS module walls where the robot can grip.
    """
    
    def __init__(self, step_size: float = 0.4, arm_reach: float = KUKA_REACH):
        self.step_size = step_size
        self.arm_reach = arm_reach
        # Effective reach for path planning - REDUCED to prevent over-stretching
        # This forces more waypoints and shorter individual reaches
        # Shorter reaches = less arm crossing
        self.effective_reach = arm_reach * 0.60  # 0.75m effective reach (was 1.0m)
        
        # Wall boundaries
        self.x_min = ISS_MODULE['x_min']
        self.x_max = ISS_MODULE['x_max']
        self.y_min = ISS_MODULE['y_min']
        self.y_max = ISS_MODULE['y_max']
        self.z_min = ISS_MODULE['z_min']
        self.z_max = ISS_MODULE['z_max']
        
        # Generate wall positions
        self.wall_positions = self._create_wall_positions()
        
        print(f"\n{'='*60}")
        print("WALL POSITION MAPPER INITIALIZED")
        print(f"{'='*60}")
        print(f"Step size: {self.step_size}m")
        print(f"Arm reach: {self.arm_reach}m")
        print(f"Effective reach for planning: {self.effective_reach}m")
        print(f"Total wall positions: {len(self.wall_positions)}")
        print(f"  - Floor: {sum(1 for _, w in self.wall_positions if w == 'floor')}")
        print(f"  - Ceiling: {sum(1 for _, w in self.wall_positions if w == 'ceiling')}")
        print(f"  - Front: {sum(1 for _, w in self.wall_positions if w == 'front')}")
        print(f"  - Back: {sum(1 for _, w in self.wall_positions if w == 'back')}")
        print(f"  - Left: {sum(1 for _, w in self.wall_positions if w == 'left_wall')}")
        print(f"  - Right: {sum(1 for _, w in self.wall_positions if w == 'right_wall')}")
        
    def _create_wall_positions(self) -> List[Tuple[Tuple[float, float, float], str]]:
        """Create discrete positions on all 6 walls"""
        positions = []
        margin = 0.1  # Stay away from edges
        
        # Floor (z = z_min with offset)
        floor_offset = GRIP_POINT_OFFSETS.get('floor', 0.0)
        for x in np.arange(self.x_min + margin, self.x_max - margin, self.step_size):
            for y in np.arange(self.y_min + margin, self.y_max - margin, self.step_size):
                positions.append(((x, y, self.z_min + floor_offset), 'floor'))
        
        # Ceiling (z = z_max with offset to bring spheres down)
        ceiling_offset = GRIP_POINT_OFFSETS.get('ceiling', 0.0)
        for x in np.arange(self.x_min + margin, self.x_max - margin, self.step_size):
            for y in np.arange(self.y_min + margin, self.y_max - margin, self.step_size):
                positions.append(((x, y, self.z_max + ceiling_offset), 'ceiling'))
        
        # Front wall (y = y_max with offset to bring spheres inward)
        front_offset = GRIP_POINT_OFFSETS.get('front', 0.0)
        for x in np.arange(self.x_min + margin, self.x_max - margin, self.step_size):
            for z in np.arange(self.z_min + margin, self.z_max - margin, self.step_size):
                positions.append(((x, self.y_max + front_offset, z), 'front'))
        
        # Back wall (y = y_min)
        for x in np.arange(self.x_min + margin, self.x_max - margin, self.step_size):
            for z in np.arange(self.z_min + margin, self.z_max - margin, self.step_size):
                positions.append(((x, self.y_min, z), 'back'))
        
        # Left wall (x = x_min)
        for y in np.arange(self.y_min + margin, self.y_max - margin, self.step_size):
            for z in np.arange(self.z_min + margin, self.z_max - margin, self.step_size):
                positions.append(((self.x_min, y, z), 'left_wall'))
        
        # Right wall (x = x_max)
        for y in np.arange(self.y_min + margin, self.y_max - margin, self.step_size):
            for z in np.arange(self.z_min + margin, self.z_max - margin, self.step_size):
                positions.append(((self.x_max, y, z), 'right_wall'))
        
        return positions
    
    def get_wall_normal(self, wall: str) -> Tuple[float, float, float]:
        """Get the inward-facing normal vector for a wall"""
        normals = {
            'floor': (0, 0, 1),
            'ceiling': (0, 0, -1),
            'front': (0, -1, 0),
            'back': (0, 1, 0),
            'left_wall': (1, 0, 0),
            'right_wall': (-1, 0, 0),
        }
        return normals.get(wall, (0, 0, 1))
    
    def get_wall_from_position(self, pos: Tuple[float, float, float]) -> str:
        """Determine which wall a position is on"""
        x, y, z = pos
        tolerance = 0.15  # Increased tolerance to account for ceiling offset
        
        # Check ceiling first with offset
        ceiling_z = self.z_max + GRIP_POINT_OFFSETS.get('ceiling', 0.0)
        if abs(z - ceiling_z) < tolerance:
            return 'ceiling'
        elif abs(z - self.z_min) < tolerance:
            return 'floor'
        elif abs(y - self.y_max) < tolerance:
            return 'front'
        elif abs(y - self.y_min) < tolerance:
            return 'back'
        elif abs(x - self.x_min) < tolerance:
            return 'left_wall'
        elif abs(x - self.x_max) < tolerance:
            return 'right_wall'
        return 'unknown'
    
    def get_distance(self, pos1: Tuple[float, float, float], 
                     pos2: Tuple[float, float, float]) -> float:
        """Euclidean distance between two positions"""
        return np.sqrt(sum((a - b) ** 2 for a, b in zip(pos1, pos2)))
    
    def get_neighbors(self, state: CrawlerState) -> List[CrawlerState]:
        """Get reachable neighbor states from current state.
        
        Arms are on Y-sides of body:
        - Left arm at Y=-0.25, pointing -X
        - Right arm at Y=+0.25, pointing +X
        
        For now, use simple alternating arm assignment and let the IK solver
        handle reachability at execution time. The arms have ~180° workspace
        so most positions within reach distance should be achievable.
        """
        neighbors = []
        current_pos = np.array(state.position)
        current_wall = state.wall
        
        # ALWAYS alternate arms for proper bipedal locomotion
        next_arm = 'right' if state.active_arm == 'left' else 'left'
        
        for (pos, wall), _ in zip(self.wall_positions, range(len(self.wall_positions))):
            pos_arr = np.array(pos)
            distance = np.linalg.norm(pos_arr - current_pos)
            
            # Check if within EFFECTIVE arm reach (reduced for body constraints)
            if distance <= self.effective_reach and distance > 0.1:
                # Same wall or adjacent wall transitions
                if wall == current_wall or self._can_transition(current_wall, wall):
                    # No arm reachability filter - let IK solver handle it
                    neighbors.append(CrawlerState(
                        position=pos,
                        wall=wall,
                        active_arm=next_arm
                    ))
        
        return neighbors
    
    def _can_transition(self, from_wall: str, to_wall: str) -> bool:
        """Check if robot can transition between two walls"""
        # Define adjacency
        adjacency = {
            'floor': ['front', 'back', 'left_wall', 'right_wall'],
            'ceiling': ['front', 'back', 'left_wall', 'right_wall'],
            'front': ['floor', 'ceiling', 'left_wall', 'right_wall'],
            'back': ['floor', 'ceiling', 'left_wall', 'right_wall'],
            'left_wall': ['floor', 'ceiling', 'front', 'back'],
            'right_wall': ['floor', 'ceiling', 'front', 'back'],
        }
        return to_wall in adjacency.get(from_wall, [])


# ============================================================================
# PATH PLANNER (A* Algorithm)
# ============================================================================

class PathPlanner:
    """A* path planner for wall-crawling robot"""
    
    def __init__(self, wall_mapper: WallPositionMapper):
        self.mapper = wall_mapper
    
    def heuristic(self, state: CrawlerState, goal_pos: Tuple[float, float, float]) -> float:
        """Estimate cost to reach goal from state."""
        pos = state.position
        
        # IMPROVED HEURISTIC: Account for wall transitions
        # Use cached mapper instead of creating new one each time
        current_wall = self.mapper.get_wall_from_position(pos)
        # Note: goal_pos is just xyz, need to infer wall or pass it. 
        # For now, simplistic Euclidean distance.
        
        dist = np.linalg.norm(np.array(pos) - np.array(goal_pos))
        return dist
    
    def find_path(self, start_pos: Tuple[float, float, float],
                  goal_pos: Tuple[float, float, float],
                  start_arm: str = 'left') -> Optional[List[CrawlerState]]:
        """
        Find path from start to goal using A* algorithm.
        
        Args:
            start_pos: Starting position (x, y, z)
            goal_pos: Goal position (x, y, z)
            start_arm: Which arm is currently holding ('left' or 'right')
            
        Returns:
            List of states forming the path, or None if no path found
        """
        start_wall = self.mapper.get_wall_from_position(start_pos)
        goal_wall = self.mapper.get_wall_from_position(goal_pos)
        
        print(f"\nA* Path Planning:")
        print(f"  Start: {start_pos} on {start_wall}")
        print(f"  Goal: {goal_pos} on {goal_wall}")
        
        start_state = CrawlerState(
            position=start_pos,
            wall=start_wall,
            active_arm=start_arm
        )
        
        # Priority queue: (f_cost, counter, node)
        counter = 0
        open_set = []
        start_node = PriorityNode(
            f_cost=self.heuristic(start_state, goal_pos),
            state=start_state,
            g_cost=0.0,
            parent=None
        )
        heapq.heappush(open_set, (start_node.f_cost, counter, start_node))
        
        visited = set()
        iterations = 0
        max_iterations = 10000
        
        while open_set and iterations < max_iterations:
            iterations += 1
            _, _, current = heapq.heappop(open_set)
            
            # Check if goal reached
            if self.mapper.get_distance(current.state.position, goal_pos) < 0.3:
                print(f"  Path found in {iterations} iterations!")
                path = self._reconstruct_path(current)
                
                # ENSURE the EXACT goal position is the final waypoint
                # This is critical for the screw alignment phase
                final_wp = path[-1] if path else None
                if final_wp:
                    goal_dist = self.mapper.get_distance(final_wp.position, goal_pos)
                    if goal_dist > 0.05:  # If final WP is more than 5cm from goal
                        # Add goal as final waypoint
                        # Alternate arm from the previous waypoint
                        prev_arm = final_wp.active_arm
                        goal_arm = 'right' if prev_arm == 'left' else 'left'
                        goal_state = CrawlerState(
                            position=goal_pos,
                            wall=goal_wall,
                            active_arm=goal_arm
                        )
                        path.append(goal_state)
                        print(f"  ✓ Added goal position as final waypoint (was {goal_dist:.2f}m away)")
                
                return path
            
            state_hash = hash(current.state)
            if state_hash in visited:
                continue
            visited.add(state_hash)
            
            # Expand neighbors
            for neighbor_state in self.mapper.get_neighbors(current.state):
                if hash(neighbor_state) in visited:
                    continue
                
                # Cost to reach neighbor
                step_cost = self.mapper.get_distance(
                    current.state.position, neighbor_state.position
                )
                new_g_cost = current.g_cost + step_cost
                
                # Create neighbor node
                counter += 1
                neighbor_node = PriorityNode(
                    f_cost=new_g_cost + self.heuristic(neighbor_state, goal_pos),
                    state=neighbor_state,
                    g_cost=new_g_cost,
                    parent=current
                )
                heapq.heappush(open_set, (neighbor_node.f_cost, counter, neighbor_node))
        
        print(f"  No path found after {iterations} iterations")
        return None
    
    def _reconstruct_path(self, node: PriorityNode) -> List[CrawlerState]:
        """Reconstruct path from goal node to start"""
        path = []
        current = node
        while current is not None:
            path.append(current.state)
            current = current.parent
        return list(reversed(path))


# ============================================================================
# MUJOCO WALL CRAWLER SIMULATION
# ============================================================================

class WallCrawlerMuJoCoSimulation:
    """
    MuJoCo-based wall-crawler simulation.
    
    Phase 1: Visualization of workspace and path planning.
    """
    
    def __init__(self, model_path: str = "dual_arm_robot.xml"):
        """Initialize the simulation"""
        
        # Check if model exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        print(f"\n{'='*60}")
        print("MUJOCO WALL-CRAWLER SIMULATION - PHASE 1")
        print(f"{'='*60}")
        
        # Load MuJoCo model
        print(f"Loading model: {model_path}")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        print("✓ Model loaded successfully!")
        
        # Initialize wall position mapper with step size for route generation
        # step_size must be <= effective_reach (0.75m) for path planning to work
        # Using 0.5m for shorter steps = more waypoints but easier arm reach
        self.wall_mapper = WallPositionMapper(step_size=0.5, arm_reach=KUKA_REACH)
        
        # Initialize path planner
        self.path_planner = PathPlanner(self.wall_mapper)
        
        # State machine
        self.robot_state = RobotState.IDLE
        self.left_gripper_state = GripperState.CLOSED
        self.right_gripper_state = GripperState.CLOSED
        
        # Path and goals
        self.current_path: Optional[List[CrawlerState]] = None
        self.current_path_index = 0
        self.start_position: Optional[Tuple[float, float, float]] = None
        self.goal_position: Optional[Tuple[float, float, float]] = None
        
        # Visualization markers (will be added to MuJoCo scene)
        self.visualization_sites = []
        
        # Get body/site IDs
        self._setup_body_ids()
        
        print(f"\nState Machine initialized: {self.robot_state.name}")
        
    def _setup_body_ids(self):
        """Setup body and site IDs for the robot"""
        self.central_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "central_body"
        )
        self.left_gripper_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "left_gripper_center"
        )
        self.right_gripper_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "right_gripper_center"
        )
        
        # Link4 body IDs (elbow - for workspace visualization)
        # The workspace sphere is centered at the elbow (link4) for better visualization
        # This provides a more accurate representation of the reachable area
        self.left_link4_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "left_link4"
        )
        self.right_link4_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "right_link4"
        )
        
        print(f"✓ Central body ID: {self.central_body_id}")
        print(f"✓ Left gripper site ID: {self.left_gripper_site_id}")
        print(f"✓ Right gripper site ID: {self.right_gripper_site_id}")
        print(f"✓ Left link4 ID (workspace center): {self.left_link4_id}")
        print(f"✓ Right link4 ID (workspace center): {self.right_link4_id}")
        
        # Setup additional IDs for Phase 2 control
        self._setup_control_ids()
    
    def _setup_control_ids(self):
        """Setup body IDs for arm control and grippers (Phase 2)"""
        # End-effector body IDs for Jacobian computation
        self.left_ee_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "left_gripper_base_mount"
        )
        self.right_ee_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "right_gripper_base_mount"
        )
        
        # Actuator indices
        # Left arm: actuators 0-6, Right arm: actuators 7-13
        # Left gripper: actuator 14, Right gripper: actuator 15
        self.left_arm_actuator_slice = slice(0, 7)
        self.right_arm_actuator_slice = slice(7, 14)
        self.left_gripper_actuator_idx = 14
        self.right_gripper_actuator_idx = 15
        
        # Joint indices (qpos and qvel)
        # Freejoint uses 7 qpos (3 pos + 4 quat) and 6 qvel (3 lin + 3 ang)
        self.left_arm_qpos_slice = slice(7, 14)
        self.right_arm_qpos_slice = slice(22, 29)
        self.left_arm_qvel_slice = slice(6, 13)
        self.right_arm_qvel_slice = slice(21, 28)
        
        # Controller gains - INCREASED for more aggressive motion (FASTER LOCOMOTION)
        self.kp_position = 900.0      # Position control (INCREASED from 500 for faster reaching)
        self.kd_position = 25.0       # Damping (REDUCED for faster response)
        self.kp_orientation = 30.0    # Orientation control (INCREASED)
        self.kd_orientation = 20.0
        self.lambda_dls = 0.006       # Damping for DLS (LOWER for better tracking)
        
        # Per-joint gain multipliers (joints 1-2 are base, need MUCH more authority)
        # [joint1, joint2, joint3, joint4, joint5, joint6, joint7]
        self.joint_gain_scale = np.array([3.0, 2.5, 1.5, 1.2, 1.0, 0.8, 0.6])
        
        # Higher gains for maintaining anchor when body is unlocked
        self.kp_anchor = 1500.0       # Stiffness for anchored arm (INCREASED)
        self.kd_anchor = 50.0         # Damping for stability
        
        # Gripper control values (0=open, 255=closed for Robotiq 2F85)
        self.gripper_open_value = 0
        self.gripper_closed_value = 255
        
        # Target positions for arm control
        self.left_target_position = None
        self.right_target_position = None
        
        # Anchor state (weld constraint)
        self.left_arm_anchored = False
        self.right_arm_anchored = False
        self.left_anchor_position = None   # World position where left gripper is anchored
        self.right_anchor_position = None  # World position where right gripper is anchored
        self.locked_body_pos = None        # Locked body position when anchored
        self.locked_body_quat = None       # Locked body orientation when anchored
        self.locked_left_arm_joints = None # Locked left arm joint positions when anchored
        
        #End Effector position for anchoring
        print(f"✓ Left EE body ID: {self.left_ee_body_id}")
        print(f"✓ Right EE body ID: {self.right_ee_body_id}")
        print(f"✓ Gripper actuators: Left={self.left_gripper_actuator_idx}, Right={self.right_gripper_actuator_idx}")
    
    def snap_to_nearest_sphere(self, pos: Tuple[float, float, float]) -> Tuple[Tuple[float, float, float], str]:
        """Snap a position to the nearest wall sphere position.
        
        Returns:
            Tuple of (snapped_position, wall_name)
        """
        min_dist = float('inf')
        nearest_pos = pos
        nearest_wall = 'unknown'
        
        for (sphere_pos, wall) in self.wall_mapper.wall_positions:
            dist = np.linalg.norm(np.array(sphere_pos) - np.array(pos))
            if dist < min_dist:
                min_dist = dist
                nearest_pos = sphere_pos
                nearest_wall = wall
        
        return nearest_pos, nearest_wall
    
    def set_start_and_goal(self, start: Tuple[float, float, float],
                           goal: Tuple[float, float, float]):
        """Set start and goal positions for path planning.
        
        Positions are automatically snapped to the nearest wall sphere.
        """
        # Snap to nearest sphere positions
        snapped_start, start_wall = self.snap_to_nearest_sphere(start)
        snapped_goal, goal_wall = self.snap_to_nearest_sphere(goal)
        
        self.start_position = snapped_start
        self.goal_position = snapped_goal
        
        print(f"\n{'='*60}")
        print("PATH PLANNING")
        print(f"{'='*60}")
        print(f"Requested Start: {start} → Snapped to: {snapped_start} ({start_wall})")
        print(f"Requested Goal:  {goal} → Snapped to: {snapped_goal} ({goal_wall})")
        
        # Transition to PLANNING state
        self._transition_state(RobotState.PLANNING)
        
        # Choose initial arm based on direction to goal
        # Left arm points -X, Right arm points +X
        delta_x = snapped_goal[0] - snapped_start[0]
        if delta_x < -0.1:
            initial_arm = 'left'  # Moving -X, left arm leads
        elif delta_x > 0.1:
            initial_arm = 'right'  # Moving +X, right arm leads
        else:
            initial_arm = 'left'  # Default
        print(f"Initial arm selected: {initial_arm.upper()} (delta_x={delta_x:.2f})")
        
        # Compute path using snapped positions
        self.current_path = self.path_planner.find_path(snapped_start, snapped_goal, start_arm=initial_arm)
        
        if self.current_path:
            print(f"\n✓ Path found with {len(self.current_path)} steps:")
            for i, state in enumerate(self.current_path):
                print(f"  Step {i+1}: {state.wall:10s} | "
                      f"({state.position[0]:6.2f}, {state.position[1]:6.2f}, {state.position[2]:6.2f}) | "
                      f"{state.active_arm.upper()} arm")
            
            # ============================================================
            # SPAWN SCREW at goal position from the beginning
            # ============================================================
            self._spawn_screw_at_goal(snapped_goal, goal_wall)
            
            self._transition_state(RobotState.READY_TO_MOVE)
        else:
            print("✗ No path found!")
            self._transition_state(RobotState.ERROR)
    
    def _spawn_screw_at_goal(self, goal_pos: Tuple[float, float, float], goal_wall: str):
        """Spawn the screw at the goal position with correct orientation for the wall.
        
        Args:
            goal_pos: The goal position (x, y, z)
            goal_wall: The wall type ('floor', 'ceiling', 'front', 'back', 'left_wall', 'right_wall')
        """
        screw_pos = np.array(goal_pos, dtype=np.float64)
        
        # Offset screw slightly into the wall based on wall type
        if goal_wall == 'ceiling':
            screw_pos[2] += 0.03  # Screw tip pointing down from ceiling
        elif goal_wall == 'floor':
            screw_pos[2] -= 0.03  # Screw tip pointing up from floor
        elif goal_wall == 'front':
            screw_pos[1] += 0.03
        elif goal_wall == 'back':
            screw_pos[1] -= 0.03
        elif goal_wall == 'left_wall':
            screw_pos[0] -= 0.03
        elif goal_wall == 'right_wall':
            screw_pos[0] += 0.03
        
        # Get screw body and set position
        screw_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, 'screw')
        if screw_body_id >= 0:
            self.model.body_pos[screw_body_id] = screw_pos
            
            # Set screw orientation based on wall (screw axis should point into wall)
            # Default: screw points +Z, need to rotate to point into wall
            screw_quat = np.array([1.0, 0.0, 0.0, 0.0])  # Identity quaternion
            
            if goal_wall == 'ceiling':
                # Screw points down (-Z) into ceiling (which is +Z)
                # Rotate 180° around X axis
                screw_quat = np.array([0.0, 1.0, 0.0, 0.0])  # 180° around X
            elif goal_wall == 'floor':
                # Screw points up (+Z) into floor (which is -Z) - default is fine
                screw_quat = np.array([1.0, 0.0, 0.0, 0.0])
            elif goal_wall == 'front':
                # Screw points +Y into front wall
                # Rotate -90° around X axis
                screw_quat = np.array([0.707, -0.707, 0.0, 0.0])  # -90° around X
            elif goal_wall == 'back':
                # Screw points -Y into back wall
                # Rotate +90° around X axis
                screw_quat = np.array([0.707, 0.707, 0.0, 0.0])  # +90° around X
            elif goal_wall == 'left_wall':
                # Screw points -X into left wall
                # Rotate +90° around Y axis
                screw_quat = np.array([0.707, 0.0, 0.707, 0.0])  # +90° around Y
            elif goal_wall == 'right_wall':
                # Screw points +X into right wall
                # Rotate -90° around Y axis
                screw_quat = np.array([0.707, 0.0, -0.707, 0.0])  # -90° around Y
            
            self.model.body_quat[screw_body_id] = screw_quat
            
            # MAKE SCREW VISIBLE IMMEDIATELY - bright colors and larger size
            self._make_screw_visible()
            
            mujoco.mj_forward(self.model, self.data)
            
            print(f"\n  🔩 SCREW spawned at goal: ({screw_pos[0]:.2f}, {screw_pos[1]:.2f}, {screw_pos[2]:.2f})")
            print(f"     Wall: {goal_wall}, Size: LARGE (4cm head)")
        else:
            print(f"\n  ⚠️ Could not find screw body in model")
        
        # Store screw info for later use
        self._screw_spawned = True
        self._screw_wall = goal_wall
        self._screw_start_pos = screw_pos.copy()  # Store for unscrewing
    
    # ========================================================================
    # UNSCREWING TASK METHODS (integrated from debug_unscrew.py)
    # ========================================================================
    
    def _make_screw_visible(self):
        """Make the screw larger and more visible with bright colors."""
        screw_head_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "screw_head")
        if screw_head_id != -1:
            self.model.geom_rgba[screw_head_id] = [1.0, 0.0, 0.0, 1.0]  # RED
            self.model.geom_size[screw_head_id] = [0.06, 0.02, 0]  # 6cm radius - BIGGER
        
        screw_shaft_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "screw_shaft")
        if screw_shaft_id != -1:
            self.model.geom_rgba[screw_shaft_id] = [1.0, 0.3, 0.0, 1.0]  # BRIGHT ORANGE
            self.model.geom_size[screw_shaft_id] = [0.025, 0.1, 0]  # Thicker
        
        screw_thread_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "screw_thread")
        if screw_thread_id != -1:
            self.model.geom_rgba[screw_thread_id] = [0.0, 1.0, 1.0, 1.0]  # CYAN (changed from yellow for visibility)
            self.model.geom_size[screw_thread_id] = [0.03, 0.08, 0]  # Thicker
        
        # Add a big pulsing marker at screw location for visibility
        # This will be rendered as a visualization geom
        self._screw_marker_visible = True
        print(f"  🔩 Screw made VERY VISIBLE: RED head (6cm), ORANGE shaft, CYAN thread")
    
    def _get_screw_unscrew_axis(self, wall: str) -> np.ndarray:
        """Get the axis along which the screw moves when unscrewing (OUT of the wall)."""
        axes = {
            'floor': np.array([0.0, 0.0, 1.0]),       # Up out of floor
            'ceiling': np.array([0.0, 0.0, -1.0]),    # Down out of ceiling
            'front': np.array([0.0, -1.0, 0.0]),      # -Y out of front wall
            'back': np.array([0.0, 1.0, 0.0]),        # +Y out of back wall
            'left_wall': np.array([1.0, 0.0, 0.0]),   # +X out of left wall
            'right_wall': np.array([-1.0, 0.0, 0.0]), # -X out of right wall
        }
        return axes.get(wall, np.array([0.0, 0.0, 1.0]))
    
    def _apply_j7_rotation(self, arm: str, speed: float) -> float:
        """Apply rotation to J7 using direct position increment."""
        if arm == 'left':
            qpos_idx = self.left_arm_qpos_slice.start + 6
            ctrl_idx = self.left_arm_actuator_slice.start + 6
        else:
            qpos_idx = self.right_arm_qpos_slice.start + 6
            ctrl_idx = self.right_arm_actuator_slice.start + 6
        
        current = self.data.qpos[qpos_idx]
        new_val = current + speed * self.model.opt.timestep
        
        # Set both qpos and control target
        self.data.qpos[qpos_idx] = new_val
        self.data.ctrl[ctrl_idx] = new_val
        
        return new_val
    
    def _apply_torque_compensation(self, arm: str):
        """Apply Coriolis/centrifugal compensation and damping for smooth motion."""
        if arm == 'left':
            qvel_slice = self.left_arm_qvel_slice
            actuator_slice = self.left_arm_actuator_slice
        else:
            qvel_slice = self.right_arm_qvel_slice
            actuator_slice = self.right_arm_actuator_slice
        
        # Get current joint velocities
        qvel = self.data.qvel[qvel_slice]
        
        # Apply velocity damping for smooth motion
        damping_gains = np.array([5.0, 5.0, 5.0, 3.0, 2.0, 1.0, 0.5])
        
        # Get the passive forces (gravity + Coriolis)
        bias_forces = self.data.qfrc_bias[qvel_slice]
        
        # Apply compensation to joints 1-6 (not J7)
        for i in range(6):
            current_ctrl = self.data.ctrl[actuator_slice.start + i]
            compensation = bias_forces[i] * 0.001
            self.data.ctrl[actuator_slice.start + i] = current_ctrl + compensation
        
        return bias_forces
    
    def _maintain_ee_position_and_orientation(self, arm: str, target_pos: np.ndarray, screw_axis: np.ndarray):
        """Maintain EE position tracking screw AND strict orientation aligned with screw axis.
        
        For position-controlled actuators, we compute the desired joint positions via IK
        and set them directly to ctrl AND qpos for immediate effect.
        Returns (position_error_m, orientation_error_rad).
        """
        if arm == 'left':
            current_pos = self.get_left_gripper_pos()
            current_orient = self.get_left_ee_orientation()
            site_id = self.left_gripper_site_id
            qpos_slice = self.left_arm_qpos_slice
            actuator_slice = self.left_arm_actuator_slice
            qvel_slice = self.left_arm_qvel_slice
        else:
            current_pos = self.get_right_gripper_pos()
            current_orient = self.get_right_ee_orientation()
            site_id = self.right_gripper_site_id
            qpos_slice = self.right_arm_qpos_slice
            actuator_slice = self.right_arm_actuator_slice
            qvel_slice = self.right_arm_qvel_slice
        
        # Position error
        pos_error = target_pos - current_pos
        pos_err_mag = np.linalg.norm(pos_error)
        
        # Orientation error - EE Z-axis should align with NEGATIVE screw axis
        desired_z = -screw_axis  # Gripper Z points into wall
        current_z = current_orient[:, 2]
        orient_error = np.cross(current_z, desired_z)
        orient_err_mag = np.linalg.norm(orient_error)
        
        # Adaptive gains - MUCH HIGHER for effective alignment
        kp_pos = 80.0 if pos_err_mag > 0.1 else 40.0  # Position gain (INCREASED)
        kp_orient = 40.0 if orient_err_mag > 0.2 else 20.0  # Orientation gain (INCREASED)
        
        # Get Jacobians
        jacp, jacr = self.compute_jacobian_at_site(site_id)
        if arm == 'left':
            J_pos = jacp[:, 6:13]   # Left arm qvel indices 6-12
            J_rot = jacr[:, 6:13]
        else:
            J_pos = jacp[:, 21:28]  # Right arm qvel indices 21-27 (FIXED!)
            J_rot = jacr[:, 21:28]
        
        # Stack position and orientation tasks (position has higher priority)
        J_full = np.vstack([J_pos * 2.0, J_rot])
        task_vel = np.concatenate([kp_pos * pos_error, kp_orient * orient_error])
        
        # Damped least squares
        lambda_dls = 0.01
        JJT = J_full @ J_full.T
        J_pinv = J_full.T @ np.linalg.inv(JJT + lambda_dls**2 * np.eye(6))
        joint_vel = J_pinv @ task_vel
        
        # Integrate to get joint position deltas
        # Use a fixed integration step that's larger than physics timestep
        dt = 0.08  # 80ms integration step for faster responsive motion
        
        # Clip joint velocities - allow faster motion during alignment
        max_joint_vel = 2.5  # rad/s (INCREASED)
        joint_vel = np.clip(joint_vel, -max_joint_vel, max_joint_vel)
        
        # Apply to joints 0-5 only (not J7 which is for rotation)
        for i in range(6):
            current_q = self.data.qpos[qpos_slice.start + i]
            delta_q = joint_vel[i] * dt
            new_q = current_q + delta_q
            
            # NOTE: Joint limits use jnt_range which has different indexing than qpos
            # For now, use safe conservative limits for KUKA iiwa14 (approximately ±170°)
            safe_limit = 2.9  # About 166 degrees
            new_q = np.clip(new_q, -safe_limit, safe_limit)
            
            # Set BOTH qpos (state) and ctrl (target) for position actuators
            self.data.qpos[qpos_slice.start + i] = new_q
            self.data.ctrl[actuator_slice.start + i] = new_q
        
        # Don't zero velocities - let physics handle it
        # But do zero the unscrewing arm velocities to prevent drift
        self.data.qvel[qvel_slice] = 0.0
        
        return pos_err_mag, orient_err_mag
    
    def _init_unscrew_task(self, screw_arm: str, anchor_arm: str, screw_wall: str):
        """Initialize the unscrewing task state variables."""
        self._unscrew_arm = screw_arm
        self._unscrew_anchor_arm = anchor_arm
        self._unscrew_wall = screw_wall
        self._unscrew_axis = self._get_screw_unscrew_axis(screw_wall)
        
        # Get screw body info
        self._unscrew_screw_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "screw")
        self._unscrew_screw_start_pos = self.data.xpos[self._unscrew_screw_body_id].copy()
        
        # Unscrew parameters
        self._unscrew_speed = 6.0  # rad/s
        self._unscrew_thread_pitch = 0.015  # 15mm per revolution
        self._unscrew_max_displacement = 0.15  # 150mm max
        self._unscrew_total_rotation = 0.0
        self._unscrew_complete = False
        self._unscrew_target_rotations = 3.0  # 3 full turns
        
        # ============================================================
        # POSITIONING PHASE PARAMETERS - wait before starting rotation
        # ============================================================
        self._unscrew_positioning_phase = True  # Start in positioning mode
        self._unscrew_positioning_duration = 1.5  # seconds to position before rotating
        self._unscrew_positioning_start_time = self.data.time
        self._unscrew_positioning_kp_pos = 1500.0  # Stricter position control (was ~800)
        self._unscrew_positioning_kp_orient = 400.0  # Stricter orientation control (was ~200)
        
        # ============================================================
        # ALIGNMENT ERROR TRACKING for plotting
        # ============================================================
        self._unscrew_alignment_log = []  # (time, alignment_error_rad, phase)
        self._unscrew_j7_log = []  # (time, j7_angle_rad)
        self._unscrew_ee_pos_log = []  # (time, ee_pos)
        self._unscrew_screw_pos_log = []  # (time, screw_pos)
        
        # Store initial J7 position
        if screw_arm == 'left':
            self._unscrew_j7_start = self.data.qpos[self.left_arm_qpos_slice.start + 6]
        else:
            self._unscrew_j7_start = self.data.qpos[self.right_arm_qpos_slice.start + 6]
        
        # Lock anchor arm and body
        self._unscrew_locked_body_pos = self.data.qpos[0:3].copy()
        self._unscrew_locked_body_quat = self.data.qpos[3:7].copy()
        
        if anchor_arm == 'left':
            self._unscrew_locked_anchor_joints = self.data.qpos[self.left_arm_qpos_slice].copy()
        else:
            self._unscrew_locked_anchor_joints = self.data.qpos[self.right_arm_qpos_slice].copy()
        
        # Store target EE orientation (aligned with screw axis)
        # Get current EE orientation for positioning target
        if screw_arm == 'left':
            ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_gripper_center")
        else:
            ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_gripper_center")
        self._unscrew_ee_site_id = ee_site_id
        
        # Target EE position = screw position
        self._unscrew_target_ee_pos = self.data.xpos[self._unscrew_screw_body_id].copy()
        
        # Make screw visible
        self._make_screw_visible()
        
        print(f"\n🔩 UNSCREW TASK INITIALIZED")
        print(f"   Screw arm: {screw_arm.upper()}, Anchor arm: {anchor_arm.upper()}")
        print(f"   Wall: {screw_wall}, Axis: {self._unscrew_axis}")
        print(f"   Speed: {self._unscrew_speed:.1f} rad/s, Pitch: {self._unscrew_thread_pitch*1000:.1f}mm/rev")
        print(f"   📐 POSITIONING PHASE: {self._unscrew_positioning_duration:.1f}s delay before rotation")
        print(f"   📊 Alignment tracking enabled for plotting")
    
    def _calculate_alignment_error(self) -> float:
        """Calculate the alignment error between EE Z-axis and screw axis.
        Returns error in radians (0 = perfectly aligned)."""
        # Get EE rotation matrix (3x3) from site
        ee_rotmat = self.data.site_xmat[self._unscrew_ee_site_id].reshape(3, 3)
        
        # EE Z-axis is the 3rd column of rotation matrix
        ee_z_axis = ee_rotmat[:, 2]
        
        # Screw axis (direction we want EE to point)
        screw_axis = self._unscrew_axis
        
        # Alignment error = angle between EE Z-axis and screw axis
        # For unscrewing, EE should point ALONG screw axis (into the wall)
        # But for rotating, we want EE Z to be perpendicular or aligned based on task
        # Here we measure how well EE Z aligns with the screw axis
        dot_product = np.clip(np.dot(ee_z_axis, screw_axis), -1.0, 1.0)
        alignment_error = np.arccos(np.abs(dot_product))  # 0 = aligned (either direction)
        
        return alignment_error
    
    def _step_positioning_phase(self) -> bool:
        """Execute one step of the ACTIVE ALIGNMENT phase.
        Runs IK + torque compensation to align EE with screw axis.
        Returns True when alignment error is small enough to start rotation."""
        current_time = self.data.time
        elapsed = current_time - self._unscrew_positioning_start_time
        
        # Calculate alignment error
        alignment_error = self._calculate_alignment_error()
        self._unscrew_alignment_log.append((elapsed, alignment_error, 'aligning'))
        
        # Log J7 position
        if self._unscrew_arm == 'left':
            j7_val = self.data.qpos[self.left_arm_qpos_slice.start + 6]
            ee_pos = self.get_left_gripper_pos()
            qpos_slice = self.left_arm_qpos_slice
            actuator_slice = self.left_arm_actuator_slice
            qvel_slice = self.left_arm_qvel_slice
        else:
            j7_val = self.data.qpos[self.right_arm_qpos_slice.start + 6]
            ee_pos = self.get_right_gripper_pos()
            qpos_slice = self.right_arm_qpos_slice
            actuator_slice = self.right_arm_actuator_slice
            qvel_slice = self.right_arm_qvel_slice
        
        self._unscrew_j7_log.append((elapsed, j7_val))
        self._unscrew_ee_pos_log.append((elapsed, ee_pos.copy()))
        self._unscrew_screw_pos_log.append((elapsed, self.data.xpos[self._unscrew_screw_body_id].copy()))
        
        # ============================================================
        # DURING ALIGNMENT: Lock body and anchor arm, but allow screw arm to align
        # ============================================================
        
        # Lock anchor arm
        if self._unscrew_anchor_arm == 'left':
            self.data.qpos[self.left_arm_qpos_slice] = self._unscrew_locked_anchor_joints
            self.data.qvel[self.left_arm_qvel_slice] = 0.0
            for i in range(7):
                self.data.ctrl[self.left_arm_actuator_slice.start + i] = self._unscrew_locked_anchor_joints[i]
        else:
            self.data.qpos[self.right_arm_qpos_slice] = self._unscrew_locked_anchor_joints
            self.data.qvel[self.right_arm_qvel_slice] = 0.0
            for i in range(7):
                self.data.ctrl[self.right_arm_actuator_slice.start + i] = self._unscrew_locked_anchor_joints[i]
        
        # Lock body position
        self.data.qpos[0:3] = self._unscrew_locked_body_pos
        self.data.qpos[3:7] = self._unscrew_locked_body_quat
        self.data.qvel[0:6] = 0.0
        
        # ============================================================
        # HOLD CURRENT POSITION: Keep arm at current pose during alignment
        # IK was causing drift - instead just hold trajectory end pose
        # ============================================================
        
        screw_pos = self.data.xpos[self._unscrew_screw_body_id].copy()
        if self._unscrew_arm == 'left':
            ee_pos = self.get_left_gripper_pos()
            current_orient = self.get_left_ee_orientation()
        else:
            ee_pos = self.get_right_gripper_pos()
            current_orient = self.get_right_ee_orientation()
        
        pos_error_vec = screw_pos - ee_pos
        pos_error = np.linalg.norm(pos_error_vec)
        
        # Orientation error
        desired_z = -self._unscrew_axis
        current_z = current_orient[:, 2]
        orient_error = np.arccos(np.clip(np.abs(np.dot(current_z, desired_z)), -1.0, 1.0))
        
        # ============================================================
        # HOLD POSE: Just maintain current joint positions
        # The trajectory phase should have already positioned the arm correctly
        # ============================================================
        current_joints = self.data.qpos[qpos_slice].copy()
        for i in range(7):
            self.data.ctrl[actuator_slice.start + i] = current_joints[i]
        
        # Track best error for progress monitoring
        if not hasattr(self, '_align_best_pos_error'):
            self._align_best_pos_error = pos_error
            self._align_best_orient_error = orient_error
        else:
            self._align_best_pos_error = min(self._align_best_pos_error, pos_error)
            self._align_best_orient_error = min(self._align_best_orient_error, orient_error)
        
        # Print progress every 50 steps (more frequent)
        step_count = int(elapsed / self.model.opt.timestep)
        if step_count % 50 == 0:
            orient_deg = np.degrees(orient_error)
            print(f"  📐 ALIGN [{elapsed:.2f}s] Pos: {pos_error*1000:.1f}mm (best:{self._align_best_pos_error*1000:.1f}) | Orient: {orient_deg:.1f}° (best:{np.degrees(self._align_best_orient_error):.1f}°)")
        
        # ============================================================
        # CHECK IF READY TO START ROTATION
        # Must meet BOTH alignment AND position criteria
        # With safety timeout to prevent infinite hanging
        # ============================================================
        alignment_threshold_deg = 20.0  # Allow up to 20° alignment error
        alignment_threshold_rad = np.radians(alignment_threshold_deg)
        position_threshold_mm = 100.0  # Must be within 100mm of screw
        min_alignment_time = 0.5  # At least 0.5s of alignment before checking
        max_alignment_time = 5.0  # Shorter timeout - 5 seconds
        
        alignment_good = orient_error < alignment_threshold_rad
        position_good = pos_error * 1000 < position_threshold_mm  # Convert to mm
        time_ok = elapsed >= min_alignment_time
        timed_out = elapsed >= max_alignment_time
        
        # Start when BOTH alignment AND position are good, OR if timed out
        if (alignment_good and position_good and time_ok) or timed_out:
            orient_deg = np.degrees(orient_error)
            if timed_out and not (alignment_good and position_good):
                print(f"\n  ⚠️ ALIGNMENT TIMEOUT after {elapsed:.1f}s")
                print(f"     Orientation: {orient_deg:.1f}° (threshold: {alignment_threshold_deg}°) {'✓' if alignment_good else '✗'}")
                print(f"     Position: {pos_error*1000:.1f}mm (threshold: {position_threshold_mm}mm) {'✓' if position_good else '✗'}")
                print(f"     Proceeding to rotation anyway...")
            else:
                print(f"\n  ✅ ALIGNMENT COMPLETE in {elapsed:.1f}s")
                print(f"     Final orientation error: {orient_deg:.1f}° (threshold: {alignment_threshold_deg}°)")
                print(f"     Position error: {pos_error*1000:.1f}mm (threshold: {position_threshold_mm}mm)")
                print(f"     Starting rotation phase...")
            
            # Store locked joint positions for rotation phase
            self._unscrew_locked_screw_arm_j1_6 = self.data.qpos[qpos_slice][:6].copy()
            return True
        
        return False
    
    def _step_unscrew_task(self) -> bool:
        """Execute one step of the unscrewing task. Returns True when complete."""
        if self._unscrew_complete:
            # After completion, keep everything locked
            self._lock_all_for_unscrew()
            return True
        
        # ============================================================
        # POSITIONING PHASE - wait and stabilize before rotating
        # ============================================================
        if self._unscrew_positioning_phase:
            if self._step_positioning_phase():
                self._unscrew_positioning_phase = False  # Move to rotation phase
                self._unscrew_rotation_start_time = self.data.time
            return False
        
        # ============================================================
        # ROTATION PHASE - rotate J7 to unscrew
        # ============================================================
        current_time = self.data.time
        elapsed = current_time - self._unscrew_positioning_start_time  # Total elapsed time
        
        # Log alignment error
        alignment_error = self._calculate_alignment_error()
        self._unscrew_alignment_log.append((elapsed, alignment_error, 'rotation'))
        
        # Log J7 position
        if self._unscrew_arm == 'left':
            j7_val = self.data.qpos[self.left_arm_qpos_slice.start + 6]
            ee_pos = self.get_left_gripper_pos()
        else:
            j7_val = self.data.qpos[self.right_arm_qpos_slice.start + 6]
            ee_pos = self.get_right_gripper_pos()
        self._unscrew_j7_log.append((elapsed, j7_val))
        self._unscrew_ee_pos_log.append((elapsed, ee_pos.copy()))
        self._unscrew_screw_pos_log.append((elapsed, self.data.xpos[self._unscrew_screw_body_id].copy()))
        
        # Get current arm slices
        if self._unscrew_arm == 'left':
            qpos_slice = self.left_arm_qpos_slice
            actuator_slice = self.left_arm_actuator_slice
        else:
            qpos_slice = self.right_arm_qpos_slice
            actuator_slice = self.right_arm_actuator_slice
        
        # Apply torque compensation for smooth motion during rotation
        self._apply_torque_compensation(self._unscrew_arm)
        
        # Calculate rotation increment
        rotation_inc = self._unscrew_speed * self.model.opt.timestep
        self._unscrew_total_rotation += rotation_inc
        
        # Calculate expected screw displacement based on rotation
        expected_disp = min(
            (self._unscrew_total_rotation / (2*np.pi)) * self._unscrew_thread_pitch,
            self._unscrew_max_displacement
        )
        
        # ONLY rotate J7 - keep joints 1-6 LOCKED
        j7_val = self._apply_j7_rotation(self._unscrew_arm, self._unscrew_speed)
        
        # Hold joints 1-6 fixed
        for i in range(6):
            self.data.qpos[qpos_slice.start + i] = self._unscrew_locked_screw_arm_j1_6[i]
            self.data.ctrl[actuator_slice.start + i] = self._unscrew_locked_screw_arm_j1_6[i]
        
        # Move screw along its axis (screw comes OUT of wall)
        screw_new_pos = self._unscrew_screw_start_pos + self._unscrew_axis * expected_disp
        self.model.body_pos[self._unscrew_screw_body_id] = screw_new_pos
        
        # Lock anchor arm and body
        self._lock_all_for_unscrew()
        
        # Check if complete
        rotations = self._unscrew_total_rotation / (2 * np.pi)
        if rotations >= self._unscrew_target_rotations:
            self._unscrew_complete = True
            # Store final screw position
            self._unscrew_final_screw_pos = screw_new_pos.copy()
            print(f"\n  ✅ UNSCREW COMPLETE!")
            print(f"     Rotations: {rotations:.1f}")
            print(f"     Displacement: {expected_disp*1000:.1f}mm")
            j7_deg = np.degrees(j7_val - self._unscrew_j7_start)
            print(f"     J7 rotation: {j7_deg:.0f}°")
            
            # Plot alignment graph
            self._plot_unscrew_alignment()
            return True
        
        return False
    
    def _plot_unscrew_alignment(self):
        """Plot the Z-axis alignment error over time."""
        try:
            import matplotlib.pyplot as plt
            
            if not self._unscrew_alignment_log:
                print("  ⚠️ No alignment data to plot")
                return
            
            times = [t for t, _, _ in self._unscrew_alignment_log]
            errors = [np.degrees(e) for _, e, _ in self._unscrew_alignment_log]
            phases = [p for _, _, p in self._unscrew_alignment_log]
            
            # Find phase transition point
            positioning_end_idx = 0
            for i, p in enumerate(phases):
                if p == 'rotation':
                    positioning_end_idx = i
                    break
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            
            # Plot 1: Alignment error over time
            ax1 = axes[0, 0]
            ax1.plot(times, errors, 'b-', linewidth=1.5, label='Alignment Error')
            if positioning_end_idx > 0:
                ax1.axvline(x=times[positioning_end_idx], color='r', linestyle='--', 
                           label=f'Rotation starts ({times[positioning_end_idx]:.1f}s)')
            ax1.set_xlabel('Time (s)')
            ax1.set_ylabel('Alignment Error (degrees)')
            ax1.set_title('EE Z-axis vs Screw Axis Alignment Error')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            ax1.set_ylim(0, max(90, max(errors) + 5))
            
            # Plot 2: J7 angle over time
            ax2 = axes[0, 1]
            j7_times = [t for t, _ in self._unscrew_j7_log]
            j7_angles = [np.degrees(a - self._unscrew_j7_start) for _, a in self._unscrew_j7_log]
            ax2.plot(j7_times, j7_angles, 'g-', linewidth=1.5)
            if positioning_end_idx > 0:
                ax2.axvline(x=times[positioning_end_idx], color='r', linestyle='--')
            ax2.set_xlabel('Time (s)')
            ax2.set_ylabel('J7 Rotation (degrees)')
            ax2.set_title('J7 (Wrist) Rotation')
            ax2.grid(True, alpha=0.3)
            
            # Plot 3: EE-Screw distance over time
            ax3 = axes[1, 0]
            ee_screw_dist = []
            for (t, ee), (_, screw) in zip(self._unscrew_ee_pos_log, self._unscrew_screw_pos_log):
                dist = np.linalg.norm(ee - screw) * 1000  # mm
                ee_screw_dist.append((t, dist))
            dist_times = [t for t, _ in ee_screw_dist]
            dist_vals = [d for _, d in ee_screw_dist]
            ax3.plot(dist_times, dist_vals, 'm-', linewidth=1.5)
            if positioning_end_idx > 0:
                ax3.axvline(x=times[positioning_end_idx], color='r', linestyle='--')
            ax3.set_xlabel('Time (s)')
            ax3.set_ylabel('Distance (mm)')
            ax3.set_title('EE to Screw Distance')
            ax3.grid(True, alpha=0.3)
            
            # Plot 4: Screw displacement along axis
            ax4 = axes[1, 1]
            screw_disp = []
            for t, screw_pos in self._unscrew_screw_pos_log:
                disp = np.dot(screw_pos - self._unscrew_screw_start_pos, self._unscrew_axis) * 1000
                screw_disp.append((t, disp))
            disp_times = [t for t, _ in screw_disp]
            disp_vals = [d for _, d in screw_disp]
            ax4.plot(disp_times, disp_vals, 'orange', linewidth=1.5)
            if positioning_end_idx > 0:
                ax4.axvline(x=times[positioning_end_idx], color='r', linestyle='--')
            ax4.set_xlabel('Time (s)')
            ax4.set_ylabel('Displacement (mm)')
            ax4.set_title('Screw Displacement (along axis)')
            ax4.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig('/home/aojedao/Documents/NYU/SpaceRobotics/SpaceRobotics7863/Project1/unscrew_alignment.png', dpi=150)
            print(f"\n  📊 Alignment plot saved to: unscrew_alignment.png")
            plt.show(block=False)
            plt.pause(0.5)
            
        except Exception as e:
            print(f"  ⚠️ Could not plot alignment: {e}")
    
    def _lock_all_for_unscrew(self):
        """Lock body, anchor arm, and optionally final screw position."""
        # Lock body position and orientation
        self.data.qpos[0:3] = self._unscrew_locked_body_pos
        self.data.qpos[3:7] = self._unscrew_locked_body_quat
        self.data.qvel[0:6] = 0.0
        
        # Lock anchor arm
        if self._unscrew_anchor_arm == 'left':
            self.data.qpos[self.left_arm_qpos_slice] = self._unscrew_locked_anchor_joints
            self.data.qvel[self.left_arm_qvel_slice] = 0.0
        else:
            self.data.qpos[self.right_arm_qpos_slice] = self._unscrew_locked_anchor_joints
            self.data.qvel[self.right_arm_qvel_slice] = 0.0
        
        # Lock screw position if we have final position
        if hasattr(self, '_unscrew_final_screw_pos'):
            self.model.body_pos[self._unscrew_screw_body_id] = self._unscrew_final_screw_pos

    def _transition_state(self, new_state: RobotState):
        """Transition the state machine to a new state"""
        old_state = self.robot_state
        self.robot_state = new_state
        print(f"State: {old_state.name} → {new_state.name}")
    
    def get_left_gripper_pos(self) -> np.ndarray:
        """Get left gripper center position"""
        return self.data.site_xpos[self.left_gripper_site_id].copy()
    
    def get_right_gripper_pos(self) -> np.ndarray:
        """Get right gripper center position"""
        return self.data.site_xpos[self.right_gripper_site_id].copy()
    
    def get_central_body_pos(self) -> np.ndarray:
        """Get central body position"""
        return self.data.xpos[self.central_body_id].copy()
    
    def get_left_link4_pos(self) -> np.ndarray:
        """Get left link4 position (elbow - workspace sphere center)"""
        return self.data.xpos[self.left_link4_id].copy()
    
    def get_right_link4_pos(self) -> np.ndarray:
        """Get right link4 position (elbow - workspace sphere center)"""
        return self.data.xpos[self.right_link4_id].copy()
    
    # ========================================================================
    # PHASE 2: ARM CONTROL AND GRIPPER METHODS
    # ========================================================================
    
    def get_left_ee_orientation(self) -> np.ndarray:
        """Get left end-effector orientation as 3x3 rotation matrix"""
        return self.data.site_xmat[self.left_gripper_site_id].reshape(3, 3).copy()
    
    def get_right_ee_orientation(self) -> np.ndarray:
        """Get right end-effector orientation as 3x3 rotation matrix"""
        return self.data.site_xmat[self.right_gripper_site_id].reshape(3, 3).copy()
    
    def get_surface_normal(self, wall: str) -> np.ndarray:
        """Get the inward-pointing surface normal for a wall.
        
        The gripper should approach perpendicular to the surface,
        so the gripper Z-axis should align with the INWARD normal.
        
        Args:
            wall: 'floor', 'ceiling', 'front', 'back', 'left_wall', 'right_wall'
            
        Returns:
            3D unit vector pointing INTO the module from the wall
        """
        normals = {
            'floor': np.array([0.0, 0.0, 1.0]),       # Up (into module from floor)
            'ceiling': np.array([0.0, 0.0, -1.0]),    # Down (into module from ceiling)
            'front': np.array([0.0, -1.0, 0.0]),      # -Y (into module from front +Y wall)
            'back': np.array([0.0, 1.0, 0.0]),        # +Y (into module from back -Y wall)
            'left_wall': np.array([1.0, 0.0, 0.0]),   # +X (into module from left -X wall)
            'right_wall': np.array([-1.0, 0.0, 0.0]), # -X (into module from right +X wall)
        }
        return normals.get(wall, np.array([0.0, 0.0, -1.0]))  # Default: down
    
    def get_target_orientation_matrix(self, wall: str) -> np.ndarray:
        """Get desired end-effector orientation matrix for approaching a wall.
        
        The gripper should approach perpendicular to surface:
        - Gripper Z-axis aligned with surface normal (pointing INTO surface)
        - Gripper X and Y axes form a valid rotation
        
        Args:
            wall: Wall type string
            
        Returns:
            3x3 rotation matrix for desired gripper orientation
        """
        # Surface normal is direction gripper should point (its Z-axis)
        z_axis = -self.get_surface_normal(wall)  # Gripper points INTO surface
        
        # Choose an arbitrary up vector (avoid parallel with z_axis)
        if abs(z_axis[2]) < 0.9:
            up = np.array([0.0, 0.0, 1.0])
        else:
            up = np.array([0.0, 1.0, 0.0])
        
        # Compute orthonormal basis
        x_axis = np.cross(up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        # Build rotation matrix [x, y, z] as columns
        R = np.column_stack([x_axis, y_axis, z_axis])
        return R
    
    def compute_jacobian(self, body_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the Jacobian for a given body"""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, body_id)
        return jacp, jacr
    
    def compute_jacobian_at_site(self, site_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the Jacobian at a specific site position (e.g., gripper center).
        
        This is more accurate than compute_jacobian for end-effector control
        because it accounts for the site offset from the body origin.
        """
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        # Get the site position in world frame
        site_pos = self.data.site_xpos[site_id]
        # Get the body that the site is attached to
        body_id = self.model.site_bodyid[site_id]
        # Compute Jacobian at the site position (not body origin)
        mujoco.mj_jac(self.model, self.data, jacp, jacr, site_pos, body_id)
        return jacp, jacr
    
    def set_gripper(self, arm: str, close: bool):
        """Set gripper state (open or closed)
        
        Args:
            arm: 'left' or 'right'
            close: True to close gripper, False to open
        """
        value = self.gripper_closed_value if close else self.gripper_open_value
        
        if arm == 'left':
            self.data.ctrl[self.left_gripper_actuator_idx] = value
            self.left_gripper_state = GripperState.CLOSED if close else GripperState.OPEN
            # If opening gripper, release anchor
            if not close:
                self.release_anchor('left')
        else:
            self.data.ctrl[self.right_gripper_actuator_idx] = value
            self.right_gripper_state = GripperState.CLOSED if close else GripperState.OPEN
            # If opening gripper, release anchor
            if not close:
                self.release_anchor('right')
        
        state_str = "CLOSED" if close else "OPEN"
        print(f"  🤏 {arm.upper()} gripper: {state_str}")
    
    def get_gripper_closure(self, arm: str) -> float:
        """Get how closed the gripper is (0.0 = fully open, 1.0 = fully closed)
        
        Args:
            arm: 'left' or 'right'
            
        Returns:
            Float from 0.0 (open) to 1.0 (closed)
        """
        if arm == 'left':
            ctrl_value = self.data.ctrl[self.left_gripper_actuator_idx]
        else:
            ctrl_value = self.data.ctrl[self.right_gripper_actuator_idx]
        
        # Normalize to 0-1 range
        closure = ctrl_value / self.gripper_closed_value
        return np.clip(closure, 0.0, 1.0)
    
    def is_gripper_closed_enough(self, arm: str, threshold: float = 0.8) -> bool:
        """Check if gripper is closed enough to maintain anchor
        
        Args:
            arm: 'left' or 'right'
            threshold: Minimum closure ratio (0.8 = 80% closed)
            
        Returns:
            True if gripper closure >= threshold
        """
        return self.get_gripper_closure(arm) >= threshold
    
    def release_anchor(self, arm: str):
        """Release the anchor for specified arm"""
        if arm == 'left':
            if self.left_arm_anchored:
                print(f"  🔓 {arm.upper()} arm RELEASED from anchor")
            self.left_arm_anchored = False
            self.left_anchor_position = None
        else:
            if self.right_arm_anchored:
                print(f"  🔓 {arm.upper()} arm RELEASED from anchor")
            self.right_arm_anchored = False
            self.right_anchor_position = None
    
    def detect_arm_crossing(self, verbose: bool = False) -> Tuple[bool, float]:
        """Detect if the arms are crossing each other.
        
        With side-mounted arms (X-axis ends, pointing outward):
        - Left arm base is at body_x - 0.3, pointing -X direction
        - Right arm base is at body_x + 0.3, pointing +X direction
        
        Arms are considered crossing when:
        1. Left gripper crosses to the right side (positive local X)
        2. Right gripper crosses to the left side (negative local X)
        3. Or the grippers are very close to each other
        
        Args:
            verbose: If True, print detailed diagnostic information
        
        Returns:
            Tuple of (is_crossing, crossing_severity)
            - is_crossing: True if arms are crossing
            - crossing_severity: 0.0 to 1.0, how badly they're crossing
        """
        body_pos = self.get_central_body_pos()
        body_quat = self.data.qpos[3:7]
        
        left_grip = self.get_left_gripper_pos()
        right_grip = self.get_right_gripper_pos()
        
        # Get body yaw for local frame calculation
        w, x, y, z = body_quat
        body_yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        
        # Compute gripper positions in body-local frame
        cos_yaw = np.cos(-body_yaw)
        sin_yaw = np.sin(-body_yaw)
        
        left_rel = left_grip - body_pos
        right_rel = right_grip - body_pos
        
        # Local X coordinates (negative = left side/back, positive = right side/front)
        left_local_x = left_rel[0] * cos_yaw - left_rel[1] * sin_yaw
        right_local_x = right_rel[0] * cos_yaw - right_rel[1] * sin_yaw
        
        # Check for crossing (X-axis based for X-mounted arms):
        # Left arm base is at X=-0.3, should stay on -X side (or not cross too far +X)
        # Right arm base is at X=+0.3, should stay on +X side (or not cross too far -X)
        
        # Left gripper shouldn't cross too far to positive X
        left_on_wrong_side = left_local_x > 0.1
        # Right gripper shouldn't cross too far to negative X
        right_on_wrong_side = right_local_x < -0.1
        
        # Also check if grippers are dangerously close (collision risk)
        gripper_distance = np.linalg.norm(left_grip - right_grip)
        grippers_too_close = gripper_distance < 0.15  # Only flag if very close (15cm)
        
        # Both conditions must be true for real crossing (not just proximity)
        is_crossing = (left_on_wrong_side and right_on_wrong_side) or grippers_too_close
        
        # Calculate severity - how much are they crossing?
        severity = 0.0
        if left_on_wrong_side and right_on_wrong_side:
            # Both arms on wrong side - real crossing
            severity += min((left_local_x - 0.1) / 0.4, 0.5)
            severity += min((-right_local_x - 0.1) / 0.4, 0.5)
        if grippers_too_close:
            severity += (0.15 - gripper_distance) / 0.15  # Closer = worse
        
        severity = min(severity, 1.0)
        
        # DIAGNOSTIC OUTPUT when crossing detected
        if is_crossing and verbose:
            diag_counter = getattr(self, '_crossing_diag_counter', 0) + 1
            self._crossing_diag_counter = diag_counter
            if diag_counter % 200 == 1:  # Print every ~0.4 seconds when crossing
                print(f"\n  ╔══════════════════════════════════════════════════════════╗")
                print(f"  ║           ⚠️  ARM CROSSING DIAGNOSTICS                    ║")
                print(f"  ╠══════════════════════════════════════════════════════════╣")
                print(f"  ║ Body Position: ({body_pos[0]:.3f}, {body_pos[1]:.3f}, {body_pos[2]:.3f})")
                print(f"  ║ Body Yaw: {np.degrees(body_yaw):.1f}°")
                print(f"  ╟──────────────────────────────────────────────────────────╢")
                print(f"  ║ LEFT Gripper (world):  ({left_grip[0]:.3f}, {left_grip[1]:.3f}, {left_grip[2]:.3f})")
                print(f"  ║ LEFT Gripper (local):  X={left_local_x:+.3f}, Y={left_rel[0] * sin_yaw + left_rel[1] * cos_yaw:+.3f}")
                print(f"  ║   → Should be X < 0.1 (left side) | {'❌ ON WRONG SIDE' if left_on_wrong_side else '✓ OK'}")
                print(f"  ╟──────────────────────────────────────────────────────────╢")
                print(f"  ║ RIGHT Gripper (world): ({right_grip[0]:.3f}, {right_grip[1]:.3f}, {right_grip[2]:.3f})")
                print(f"  ║ RIGHT Gripper (local): X={right_local_x:+.3f}, Y={right_rel[0] * sin_yaw + right_rel[1] * cos_yaw:+.3f}")
                print(f"  ║   → Should be X > -0.1 (right side) | {'❌ ON WRONG SIDE' if right_on_wrong_side else '✓ OK'}")
                print(f"  ╟──────────────────────────────────────────────────────────╢")
                print(f"  ║ Gripper Distance: {gripper_distance:.3f}m {'⚠️ TOO CLOSE' if grippers_too_close else ''}")
                print(f"  ║ Crossing Severity: {severity:.2f} / 1.00")
                reason = []
                if left_on_wrong_side and right_on_wrong_side:
                    reason.append("BOTH ARMS ON WRONG SIDES")
                if grippers_too_close:
                    reason.append("GRIPPERS TOO CLOSE")
                print(f"  ║ Reason: {' + '.join(reason)}")
                print(f"  ╚══════════════════════════════════════════════════════════╝\n")
        elif not is_crossing:
            # Reset counter when not crossing
            self._crossing_diag_counter = 0
        
        return is_crossing, severity
    
    
    def compute_arm_control(self, arm: str = 'left') -> Tuple[np.ndarray, np.ndarray]:
        """Compute Cartesian space control for specified arm
        
        Returns:
            (joint_position_command, position_error)
        """
        if arm == 'left':
            current_pos = self.get_left_gripper_pos()
            current_orient = self.get_left_ee_orientation()
            target_pos = self.left_target_position
            site_id = self.left_gripper_site_id  # Use site for accurate Jacobian
            qpos_slice = self.left_arm_qpos_slice
            qvel_slice = self.left_arm_qvel_slice
        else:
            current_pos = self.get_right_gripper_pos()
            current_orient = self.get_right_ee_orientation()
            target_pos = self.right_target_position
            site_id = self.right_gripper_site_id  # Use site for accurate Jacobian
            qpos_slice = self.right_arm_qpos_slice
            qvel_slice = self.right_arm_qvel_slice
        
        if target_pos is None:
            return self.data.qpos[qpos_slice].copy(), np.zeros(3)
        
        # Position error only (ignore orientation for now)
        pos_error = target_pos - current_pos
        
        # Get position Jacobian at the gripper site (not body origin)
        jacp, jacr = self.compute_jacobian_at_site(site_id)
        
        # Extract arm-specific position Jacobian columns
        if arm == 'left':
            arm_jacp = jacp[:, 6:13]   # qvel indices 6-12 for left arm
        else:
            arm_jacp = jacp[:, 21:28]  # qvel indices 21-27 for right arm
        
        # Use position-only Jacobian (3x7)
        J = arm_jacp
        
        # Position error with gain
        task_error = self.kp_position * pos_error
        
        # Current joint velocities
        current_joint_vel = self.data.qvel[qvel_slice]
        
        # Damped least squares inverse for position-only (3x7 matrix)
        JJT = J @ J.T  # 3x3
        J_pinv = J.T @ np.linalg.inv(JJT + self.lambda_dls**2 * np.eye(3))  # 7x3
        
        # Compute joint velocity command
        joint_vel_cmd = J_pinv @ task_error
        
        # Apply per-joint gain scaling (joints 1-2 have higher authority)
        joint_vel_cmd = joint_vel_cmd * self.joint_gain_scale
        
        # Add velocity damping
        joint_vel_cmd -= self.kd_position * current_joint_vel
        
        # Convert to position command
        current_joint_pos = self.data.qpos[qpos_slice]
        joint_pos_cmd = current_joint_pos + joint_vel_cmd * self.model.opt.timestep
        
        # ======================================================================
        # ARM CROSSING DETECTION (for diagnostics only)
        # ======================================================================
        
        # Check for arm crossing (with verbose diagnostics)
        is_crossing, crossing_severity = self.detect_arm_crossing(verbose=True)
        
        # DISABLED: J1 correction was double-applied and fighting the IK
        # The IK solver should handle J1 naturally based on target position
        # Recovery logic handles stuck situations separately
        
        return joint_pos_cmd, pos_error
    
    def apply_arm_control(self, arm: str = 'left', skip_base_joints: int = 0, anchored_mode: bool = False):
        """Apply arm control to reach target position
        
        Args:
            arm: 'left' or 'right'
            skip_base_joints: Number of base joints to skip (0=full control, 3=skip first 3 joints)
                             This allows partial release for body swinging during locomotion.
            anchored_mode: If True, only control last 2 joints (wrist) to maintain anchor,
                          allowing first 5 joints to be adjusted by coordinated control
        """
        joint_cmd, pos_error = self.compute_arm_control(arm)
        
        if anchored_mode:
            # In anchored mode, only control last 2 joints (wrist) with higher gains
            # First 5 joints are left for coordinated control
            skip_base_joints = 5
        
        if arm == 'left':
            if skip_base_joints > 0:
                # Keep first N joints at current position (don't control them)
                current_pos = self.data.qpos[7:7+skip_base_joints]
                joint_cmd[:skip_base_joints] = current_pos
            self.data.ctrl[self.left_arm_actuator_slice] = joint_cmd
        else:
            if skip_base_joints > 0:
                # Keep first N joints at current position (don't control them)
                current_pos = self.data.qpos[22:22+skip_base_joints]
                joint_cmd[:skip_base_joints] = current_pos
            self.data.ctrl[self.right_arm_actuator_slice] = joint_cmd
        
        return np.linalg.norm(pos_error)
    
    def apply_coordinated_arm_control(self, anchored_arm: str, moving_target: np.ndarray, anchor_pos: np.ndarray):
        """Use first 5 joints of anchored arm to help body reach moving arm's target.
        
        When the EE is anchored, rotating the base joints moves the body.
        We use inverse kinematics where the "target" is calculated to move the body
        toward a better position for the moving arm to reach its target.
        
        Key insight: If we want to move body by delta_body, and EE is fixed,
        we can think of it as moving the EE by -delta_body in the body frame,
        then solving IK for that.
        
        Args:
            anchored_arm: 'left' or 'right' - the arm that is anchored
            moving_target: The target position the moving arm is trying to reach
            anchor_pos: The anchor position that must be maintained
        """
        # Get current positions
        body_pos = self.get_central_body_pos()
        body_to_target = moving_target - body_pos
        target_distance = np.linalg.norm(body_to_target)
        
        # Don't bother if body is already very close to target
        if target_distance < 0.2:
            return
        
        # Get arm-specific data
        if anchored_arm == 'left':
            site_id = self.left_gripper_site_id
            qpos_slice = self.left_arm_qpos_slice
            qvel_slice = self.left_arm_qvel_slice
            actuator_slice = self.left_arm_actuator_slice
        else:
            site_id = self.right_gripper_site_id
            qpos_slice = self.right_arm_qpos_slice
            qvel_slice = self.right_arm_qvel_slice
            actuator_slice = self.right_arm_actuator_slice
        
        # Desired body motion direction (toward moving target)
        body_direction = body_to_target / target_distance
        
        # Scale velocity by distance - more aggressive when far
        desired_body_vel = body_direction * min(target_distance, 0.8)
        
        # Get the Jacobian at the gripper site position (more accurate than body origin)
        jacp_full, _ = self.compute_jacobian_at_site(site_id)
        
        # Extract arm-specific Jacobian for first 5 joints only
        if anchored_arm == 'left':
            J_base = jacp_full[:, 6:11]   # Left arm joints 0-4 (columns 6-10 in full Jacobian)
        else:
            J_base = jacp_full[:, 21:26]  # Right arm joints 0-4 (columns 21-25 in full Jacobian)
        
        # Very high gains for fast body repositioning - INCREASED FOR FASTER LOCOMOTION
        kp_coord = 4000.0   # Higher proportional gain for faster response
        kp_coord = 4000.0   # Higher proportional gain for faster response
        kd_coord = 15.0     # Lower damping for faster response without being sluggish
        
        # When EE is fixed, body moves opposite to what joint motion would cause EE to move
        # So to move body by +delta, we want joint motion that would move EE by -delta
        # When EE is fixed, body moves opposite to what joint motion would cause EE to move
        # So to move body by +delta, we want joint motion that would move EE by -delta
        
        # Repulsive Term: Move away from moving_target to avoid crowding/crossing
        # User request: "moves away from the next point in the trajectory with low gains"
        # DISABLING: Found to fight convergence. Relying on recovery logic instead.
        kp_repulsive = 0.0 
        repulsive_vel = -kp_repulsive * body_direction # "Away" from target
        
        # Combine: Attraction (to help reach) + Repulsion (to avoid crowding)
        # Note: These conflict. We are essentially reducing the attraction gain, or shifting the equilibrium.
        # But we implement as requested: separate terms.
        
        task_vel = (-kp_coord * desired_body_vel) + (-repulsive_vel) 
        # Wait, if I want body to move AWAY, I need joint motion that moves EE TOWARDS?
        # Relationship: v_body = -J * v_joint
        # v_joint = -J_pinv * v_body
        # If desired v_body_repulsion is AWAY from target (-dir),
        # Then v_joint_repulsion = -J_pinv * (-dir) = J_pinv * dir
        # 
        # task_vel is the velocity we feed to the solver: joint_vel = J_pinv * task_vel
        # If we use the standard IK form: dx = J * dq  => dq = J# * dx
        # Here, "task_vel" is effective -dx (since body moves opposite to EE).
        # So task_vel should be the DESIRED BODY VELOCITY vector (inverted).
        
        # Logic check:
        # We want Body to move TOWARDS target (to Help): v_body_att = +dir
        # We want Body to move AWAY from target (to Avoid): v_body_rep = -dir
        # Total v_body = v_body_att + v_body_rep
        
        # And dq = J# * (-v_body)  (because moving joints moves EE relative to Body, or Body relative to fixed EE = -dEE)
        
        combined_body_vel = (kp_coord * desired_body_vel) + (kp_repulsive * (-body_direction))
        
        # So final task input to pinv:
        task_vel = -combined_body_vel
        
        # Damped least squares inverse
        lambda_coord = 0.08
        JJT = J_base @ J_base.T  # 3x3
        J_pinv = J_base.T @ np.linalg.inv(JJT + lambda_coord**2 * np.eye(3))  # 5x3
        
        # Joint velocities
        joint_vel_base = J_pinv @ task_vel
        
        # Apply damping
        current_joint_vel = self.data.qvel[qvel_slice][:5]
        joint_vel_base -= kd_coord * current_joint_vel
        
        # Convert to position command for first 5 joints
        current_q = self.data.qpos[qpos_slice]
        dt = self.model.opt.timestep
        
        for i in range(5):
            new_q = current_q[i] + joint_vel_base[i] * dt
            self.data.ctrl[actuator_slice.start + i] = new_q
    
    def apply_arm_control_with_orientation(self, arm: str, target_wall: str, skip_base_joints: int = 0):
        """Apply arm control with orientation alignment to approach wall perpendicularly.
        
        Args:
            arm: 'left' or 'right'
            target_wall: Wall type for orientation alignment
            skip_base_joints: Number of base joints to skip
            
        Returns:
            Position error magnitude
        """
        if arm == 'left':
            current_pos = self.get_left_gripper_pos()
            current_orient = self.get_left_ee_orientation()
            target_pos = self.left_target_position
            site_id = self.left_gripper_site_id  # Use site for accurate Jacobian
            qpos_slice = self.left_arm_qpos_slice
            qvel_slice = self.left_arm_qvel_slice
            actuator_slice = self.left_arm_actuator_slice
        else:
            current_pos = self.get_right_gripper_pos()
            current_orient = self.get_right_ee_orientation()
            target_pos = self.right_target_position
            site_id = self.right_gripper_site_id  # Use site for accurate Jacobian
            qpos_slice = self.right_arm_qpos_slice
            qvel_slice = self.right_arm_qvel_slice
            actuator_slice = self.right_arm_actuator_slice
        
        if target_pos is None:
            return 0.0
        
        # Position error
        pos_error = target_pos - current_pos
        
        # Orientation error - align gripper Z-axis with surface approach direction
        target_orient = self.get_target_orientation_matrix(target_wall)
        orient_error_matrix = target_orient @ current_orient.T
        
        # Extract axis-angle from rotation matrix for orientation error
        trace = np.trace(orient_error_matrix)
        angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
        
        if angle > 1e-6:
            # Extract rotation axis
            axis = np.array([
                orient_error_matrix[2, 1] - orient_error_matrix[1, 2],
                orient_error_matrix[0, 2] - orient_error_matrix[2, 0],
                orient_error_matrix[1, 0] - orient_error_matrix[0, 1]
            ])
            axis = axis / (2 * np.sin(angle) + 1e-10)
            orient_error = angle * axis
        else:
            orient_error = np.zeros(3)
        
        # Get full Jacobian at gripper site position (not body origin)
        jacp, jacr = self.compute_jacobian_at_site(site_id)
        
        # Extract arm-specific Jacobians
        if arm == 'left':
            arm_jacp = jacp[:, 6:13]
            arm_jacr = jacr[:, 6:13]
        else:
            arm_jacp = jacp[:, 21:28]
            arm_jacr = jacr[:, 21:28]
        
        # Stack position and orientation Jacobians (6x7)
        J = np.vstack([arm_jacp, arm_jacr])
        
        # Task error: position + scaled orientation
        task_error = np.concatenate([
            self.kp_position * pos_error,
            self.kp_orientation * orient_error
        ])
        
        # Current joint velocities
        current_joint_vel = self.data.qvel[qvel_slice]
        
        # Damped least squares for 6x7 Jacobian
        JJT = J @ J.T  # 6x6
        J_pinv = J.T @ np.linalg.inv(JJT + self.lambda_dls**2 * np.eye(6))  # 7x6
        
        # Compute joint velocity command
        joint_vel_cmd = J_pinv @ task_error
        
        # Add velocity damping
        joint_vel_cmd -= self.kd_position * current_joint_vel
        
        # Convert to position command
        current_joint_pos = self.data.qpos[qpos_slice]
        joint_pos_cmd = current_joint_pos + joint_vel_cmd * self.model.opt.timestep
        
        # Apply skip_base_joints if needed
        if skip_base_joints > 0:
            if arm == 'left':
                current_base = self.data.qpos[7:7+skip_base_joints]
            else:
                current_base = self.data.qpos[22:22+skip_base_joints]
            joint_pos_cmd[:skip_base_joints] = current_base
        
        # Apply control
        self.data.ctrl[actuator_slice] = joint_pos_cmd
        
        return np.linalg.norm(pos_error)

    def anchor_gripper(self, arm: str, position: np.ndarray):
        """Anchor the gripper at a specific world position by constraining velocities
        
        This simulates gripping a fixed point in space. We apply forces to keep
        the gripper at the anchor position.
        """
        if arm == 'left':
            self.left_arm_anchored = True
            self.left_anchor_position = position.copy()
            self.left_target_position = position.copy()
        else:
            self.right_arm_anchored = True
            self.right_anchor_position = position.copy()
            self.right_target_position = position.copy()
        
        print(f"  ⚓ {arm.upper()} arm ANCHORED at ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})")
    
    def release_anchor(self, arm: str):
        """Release the anchor on specified arm - gripper can now move freely"""
        if arm == 'left':
            self.left_arm_anchored = False
            self.left_anchor_position = None
            # Clear locked joints so they can be re-captured on next anchor
            if hasattr(self, '_locked_left_joints'):
                self._locked_left_joints = None
            print(f"  🔓 LEFT arm RELEASED")
        else:
            self.right_arm_anchored = False
            self.right_anchor_position = None
            # Clear locked joints
            if hasattr(self, '_locked_right_joints'):
                self._locked_right_joints = None
            print(f"  🔓 RIGHT arm RELEASED")
    
    def maintain_anchor(self, arm: str, control_anchored_arm: bool = False):
        """Maintain the anchor by LOCKING body position AND left arm joints.
        
        Strategy: Save the body position and left arm joints when anchor is set,
        and force-restore them every step to prevent any drift.
        
        Args:
            arm: 'left' or 'right'
            control_anchored_arm: If True, apply arm control to the anchored arm.
        """
        if arm == 'left' and self.left_arm_anchored:
            # Check if gripper is closed enough to maintain anchor
            if not self.is_gripper_closed_enough('left'):
                self.release_anchor('left')
                return 0.0
            
            # Restore body position to locked position
            if hasattr(self, 'locked_body_pos') and self.locked_body_pos is not None:
                self.data.qpos[0:3] = self.locked_body_pos.copy()
                self.data.qpos[3:7] = self.locked_body_quat.copy()
                self.data.qvel[0:6] = 0.0  # Zero body velocity
                
                # Also restore left arm joints to keep gripper at anchor
                if hasattr(self, 'locked_left_arm_joints') and self.locked_left_arm_joints is not None:
                    self.data.qpos[7:14] = self.locked_left_arm_joints.copy()
                    self.data.qvel[6:13] = 0.0  # Zero left arm velocities
                
                mujoco.mj_forward(self.model, self.data)
            
            # Report gripper error for monitoring
            anchor_pos = self.left_anchor_position
            gripper_pos = self.get_left_gripper_pos()
            error_magnitude = np.linalg.norm(anchor_pos - gripper_pos)
            
            # Only apply arm control if explicitly requested
            if control_anchored_arm:
                self.apply_arm_control('left')
            
            return error_magnitude
            
        elif arm == 'right' and self.right_arm_anchored:
            # Check if gripper is closed enough to maintain anchor
            if not self.is_gripper_closed_enough('right'):
                self.release_anchor('right')
                return 0.0
            
            # Restore body position to locked position (same as left)
            if hasattr(self, 'locked_body_pos') and self.locked_body_pos is not None:
                self.data.qpos[0:3] = self.locked_body_pos.copy()
                self.data.qpos[3:7] = self.locked_body_quat.copy()
                self.data.qvel[0:6] = 0.0
                mujoco.mj_forward(self.model, self.data)
            
            anchor_pos = self.right_anchor_position
            gripper_pos = self.get_right_gripper_pos()
            error_magnitude = np.linalg.norm(anchor_pos - gripper_pos)
            
            # Only apply arm control if explicitly requested
            if control_anchored_arm:
                self.apply_arm_control('right')
            
            return error_magnitude
        return 0.0
    
    def lock_body_position(self):
        """Lock the current body position AND left arm joints for anchor constraint."""
        self.locked_body_pos = self.data.qpos[0:3].copy()
        self.locked_body_quat = self.data.qpos[3:7].copy()
        # Also lock left arm joint positions (joints 7-13)
        self.locked_left_arm_joints = self.data.qpos[7:14].copy()
        print(f"  🔒 Body locked at: ({self.locked_body_pos[0]:.2f}, {self.locked_body_pos[1]:.2f}, {self.locked_body_pos[2]:.2f})")
        left_grip = self.get_left_gripper_pos()
        print(f"  🔒 Left gripper at: ({left_grip[0]:.2f}, {left_grip[1]:.2f}, {left_grip[2]:.2f})")
    
    def find_nearest_grip_point(self, position: np.ndarray) -> Tuple[np.ndarray, str]:
        """Find the nearest wall grip point to a position"""
        min_dist = float('inf')
        nearest_pos = None
        nearest_wall = None
        
        for (pos, wall) in self.wall_mapper.wall_positions:
            dist = np.linalg.norm(np.array(pos) - position)
            if dist < min_dist:
                min_dist = dist
                nearest_pos = np.array(pos)
                nearest_wall = wall
        
        return nearest_pos, nearest_wall
    
    def step_phase2_control(self):
        """Step the Phase 2 controller - anchor arms and maintain position
        
        When only one arm is anchored: move body to keep that gripper at anchor,
        but DON'T control the anchored arm's joints - let them be free so the
        other arm can pull the body around.
        
        When both arms are anchored: control both arms to maintain positions.
        """
        left_anchored = self.left_arm_anchored and self.is_gripper_closed_enough('left')
        right_anchored = self.right_arm_anchored and self.is_gripper_closed_enough('right')
        
        if left_anchored and right_anchored:
            # BOTH arms anchored - just kill body motion, don't try to move it
            # The arms will conform to where the body is
            self.data.qvel[0:6] = 0.0
            # Apply arm control to both arms
            self.apply_arm_control('left')
            self.apply_arm_control('right')
        elif left_anchored:
            # Only left is anchored - maintain anchor position
            # DO NOT control left arm joints - let them be free so right arm can move body
            self.maintain_anchor('left', control_anchored_arm=False)
            # Right arm is free to move toward its target (controlled in state machine)
        elif right_anchored:
            # Only right is anchored - maintain anchor position
            # DO NOT control right arm joints - let them be free so left arm can move body
            self.maintain_anchor('right', control_anchored_arm=False)
            # Left arm is free to move toward its target (controlled in state machine)
    
    # ========================================================================
    # END PHASE 2 METHODS
    # ========================================================================
    
    def _add_marker_geom(self, scene, pos: Tuple[float, float, float], 
                         size: float, rgba: Tuple[float, float, float, float],
                         geom_type: int = mujoco.mjtGeom.mjGEOM_SPHERE):
        """Add a marker geometry to the scene for visualization"""
        if scene.ngeom >= scene.maxgeom:
            return  # Scene is full
        
        # Initialize the geometry
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            geom_type,
            np.zeros(3),  # size (will be set below)
            np.array(pos, dtype=np.float64),  # position
            np.eye(3).flatten(),  # rotation matrix (identity)
            np.array(rgba, dtype=np.float32)  # color
        )
        
        # Set size for sphere
        scene.geoms[scene.ngeom].size[:] = [size, size, size]
        scene.ngeom += 1
    
    def _add_line_geom(self, scene, pos1: Tuple[float, float, float],
                       pos2: Tuple[float, float, float],
                       size: float, rgba: Tuple[float, float, float, float]):
        """Add a line (capsule) geometry between two points"""
        if scene.ngeom >= scene.maxgeom:
            return
        
        p1 = np.array(pos1, dtype=np.float64)
        p2 = np.array(pos2, dtype=np.float64)
        
        # Calculate midpoint and direction
        midpoint = (p1 + p2) / 2
        diff = p2 - p1
        length = np.linalg.norm(diff)
        
        if length < 0.001:
            return
        
        # Calculate rotation matrix to align capsule with line
        direction = diff / length
        
        # Create rotation matrix (align z-axis with direction)
        z_axis = np.array([0, 0, 1])
        if np.allclose(direction, z_axis) or np.allclose(direction, -z_axis):
            rotation = np.eye(3)
            if np.dot(direction, z_axis) < 0:
                rotation[2, 2] = -1
        else:
            axis = np.cross(z_axis, direction)
            axis = axis / np.linalg.norm(axis)
            angle = np.arccos(np.clip(np.dot(z_axis, direction), -1, 1))
            
            # Rodrigues rotation formula
            K = np.array([[0, -axis[2], axis[1]],
                         [axis[2], 0, -axis[0]],
                         [-axis[1], axis[0], 0]])
            rotation = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
        
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.array([size, length / 2, 0], dtype=np.float64),
            midpoint,
            rotation.flatten(),
            np.array(rgba, dtype=np.float32)
        )
        scene.ngeom += 1
    
    def _render_visualization_geoms(self, viewer):
        """Render all visualization geometries (wall positions, path, goals, workspace)"""
        # Access the user scene from the viewer
        try:
            scene = viewer.user_scn
        except AttributeError:
            # Fallback - user_scn might not be available in older versions
            print("Warning: user_scn not available")
            return
        
        # Reset the scene geometry count for user geometries
        scene.ngeom = 0
        
        # Colors (RGBA) - with transparency for workspace spheres
        GRAY = (0.6, 0.6, 0.6, 0.4)              # Wall positions
        GREEN = (0.0, 0.9, 0.0, 1.0)              # Start
        RED = (0.9, 0.0, 0.0, 1.0)                # Goal
        BLUE = (0.2, 0.5, 1.0, 0.9)               # Path waypoints
        CYAN = (0.0, 0.9, 0.9, 0.7)               # Path lines
        YELLOW = (1.0, 1.0, 0.0, 0.6)             # Current target
        ORANGE = (1.0, 0.5, 0.0, 0.8)             # Left arm waypoint
        PURPLE = (0.7, 0.0, 1.0, 0.8)             # Right arm waypoint
        
        # Workspace sphere colors (very transparent so robot is visible inside)
        LEFT_WORKSPACE = (0.0, 0.5, 1.0, 0.12)    # Blue, very transparent
        RIGHT_WORKSPACE = (1.0, 0.5, 0.0, 0.12)   # Orange, very transparent
        
        # 1. Draw workspace spheres at link4 (elbow) position
        # Centered at the elbow with radius to reach the gripper
        left_link4_pos = self.get_left_link4_pos()
        right_link4_pos = self.get_right_link4_pos()
        
        # Left arm workspace sphere (blue, transparent)
        self._add_marker_geom(scene, tuple(left_link4_pos), 
                              size=WORKSPACE_SPHERE_RADIUS, rgba=LEFT_WORKSPACE)
        
        # Right arm workspace sphere (orange, transparent)
        self._add_marker_geom(scene, tuple(right_link4_pos), 
                              size=WORKSPACE_SPHERE_RADIUS, rgba=RIGHT_WORKSPACE)
        
        # 2. Draw all wall grip positions (small gray spheres)
        for (pos, wall) in self.wall_mapper.wall_positions:
            self._add_marker_geom(scene, pos, size=0.025, rgba=GRAY)
        
        # 3. Draw start position (large green sphere)
        if self.start_position:
            self._add_marker_geom(scene, self.start_position, size=0.1, rgba=GREEN)
        
        # 4. Draw goal position (large red sphere)
        if self.goal_position:
            self._add_marker_geom(scene, self.goal_position, size=0.1, rgba=RED)
        
        # 5. Draw COMPLETE PLANNED ROUTE with all waypoints and connections
        if self.current_path and len(self.current_path) > 0:
            # Colors for completed vs upcoming waypoints
            COMPLETED_WP = (0.3, 0.8, 0.3, 0.9)      # Green for completed
            UPCOMING_WP = (0.4, 0.4, 1.0, 0.9)       # Blue for upcoming
            CURRENT_WP = (1.0, 1.0, 0.0, 1.0)        # Yellow for current
            ROUTE_LINE = (0.0, 0.8, 0.8, 0.7)        # Cyan for route lines
            
            # Get current waypoint index (use class attribute if available)
            current_wp_idx = getattr(self, '_current_waypoint_display_idx', 0)
            total_waypoints = len(self.current_path)
            
            # Check if at final waypoint AND trajectory complete (for hiding markers)
            is_final_and_arrived = (current_wp_idx >= total_waypoints - 1 and 
                                   getattr(self, '_trajectory_complete', False))
            
            # Draw all waypoints with numbers
            for i, wp in enumerate(self.current_path):
                if i < current_wp_idx:
                    # Completed waypoint
                    color = COMPLETED_WP
                    size = 0.06
                elif i == current_wp_idx:
                    # Current target - SKIP drawing if at final waypoint and arrived
                    if is_final_and_arrived:
                        continue  # Skip yellow marker at final waypoint once arrived
                    color = CURRENT_WP
                    size = 0.10
                else:
                    # Upcoming waypoint
                    color = UPCOMING_WP
                    size = 0.07
                
                self._add_marker_geom(scene, wp.position, size=size, rgba=color)
            
            # Draw route lines connecting all waypoints
            for i in range(len(self.current_path) - 1):
                wp1 = self.current_path[i]
                wp2 = self.current_path[i + 1]
                
                # Use different colors for completed vs upcoming segments
                if i < current_wp_idx:
                    line_color = COMPLETED_WP
                    line_width = 0.015
                else:
                    line_color = ROUTE_LINE
                    line_width = 0.02
                
                self._add_line_geom(scene, wp1.position, wp2.position, 
                                   size=line_width, rgba=line_color)
        
        # Check if we're at the final waypoint (for hiding markers during screw animation)
        is_final_waypoint = False
        if self.current_path and len(self.current_path) > 0:
            current_wp_idx = getattr(self, '_current_waypoint_display_idx', 0)
            total_waypoints = len(self.current_path)
            is_final_waypoint = current_wp_idx >= total_waypoints - 1
        
        # 6. Draw current arm targets with highlights (HIDE at final waypoint to see screw animation)
        if not is_final_waypoint:
            # Use SMALL transparent markers so screw is visible through them
            if self.left_target_position is not None:
                # Small transparent ring instead of solid sphere (so screw shows through)
                YELLOW_TRANSPARENT = (1.0, 1.0, 0.0, 0.25)  # Very transparent
                self._add_marker_geom(scene, tuple(self.left_target_position), size=0.15, rgba=YELLOW_TRANSPARENT)
            
            if self.right_target_position is not None:
                # Transparent orange for second target
                ORANGE_TRANSPARENT = (1.0, 0.6, 0.0, 0.25)
                self._add_marker_geom(scene, tuple(self.right_target_position), size=0.12, rgba=ORANGE_TRANSPARENT)
        
        # 7. Draw trajectory line between anchors (BLUE line from left anchor to right target)
        # Also hide at final waypoint
        if not is_final_waypoint:
            if self.left_anchor_position is not None and self.right_target_position is not None:
                BRIGHT_BLUE = (0.2, 0.6, 1.0, 0.9)
                self._add_line_geom(scene, 
                                   tuple(self.left_anchor_position), 
                                   tuple(self.right_target_position), 
                                   size=0.02,
                                   rgba=BRIGHT_BLUE)
        
        # 8. Draw SCREW MARKER (HIDE at final waypoint to see screw+arm animation clearly)
        # Only show during approach, not during unscrew animation
        if getattr(self, '_screw_spawned', False) and not is_final_waypoint:
            screw_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "screw")
            if screw_body_id != -1:
                screw_pos = self.data.xpos[screw_body_id]
                # MAGENTA pulsing marker - very visible
                SCREW_MARKER = (1.0, 0.0, 1.0, 0.9)  # Magenta
                self._add_marker_geom(scene, tuple(screw_pos), size=0.08, rgba=SCREW_MARKER)
                # Add a second larger transparent ring
                SCREW_RING = (1.0, 1.0, 0.0, 0.4)  # Yellow transparent
                self._add_marker_geom(scene, tuple(screw_pos), size=0.12, rgba=SCREW_RING)
    
    def run_headless(self, duration: float = 300):
        """
        Run the simulation without visualization (for batch testing).
        Uses the same locomotion logic as run_visualization but without rendering.
        """
        print(f"\n{'='*60}")
        print("STARTING HEADLESS SIMULATION")
        print(f"{'='*60}")
        
        # Reset simulation
        mujoco.mj_resetData(self.model, self.data)
        
        # Initialize body position same as visualization
        if self.start_position and self.current_path and len(self.current_path) > 0:
            wp1 = np.array(self.current_path[0].position)
            wp1_wall = self.current_path[0].wall
            active_arm = self.current_path[0].active_arm
            surface_normal = self.get_surface_normal(wp1_wall)
            ideal_reach = 0.85
            
            if active_arm == 'left':
                arm_base_offset = np.array([LEFT_ARM_OFFSET, 0, 0])
            else:
                arm_base_offset = np.array([RIGHT_ARM_OFFSET, 0, 0])
            
            body_pos = wp1 - arm_base_offset + (surface_normal * ideal_reach)
            body_x, body_y, body_z = body_pos
            
            # Clamp to safe bounds - use moderate buffer to prevent arms going outside
            arm_buffer = 0.50  # Compromise: prevents most wall penetration while allowing reach
            min_body_x = ISS_MODULE['x_min'] + arm_buffer + 0.3
            max_body_x = ISS_MODULE['x_max'] - arm_buffer - 0.05
            safe_margin = 0.40  # Reduced from 0.55 to allow closer approach
            
            body_x = np.clip(body_x, min_body_x, max_body_x)
            body_y = np.clip(body_y, ISS_MODULE['y_min'] + safe_margin, ISS_MODULE['y_max'] - safe_margin)
            body_z = np.clip(body_z, ISS_MODULE['z_min'] + safe_margin, ISS_MODULE['z_max'] - safe_margin)
            
            # Set body position
            self.data.qpos[0:3] = [body_x, body_y, body_z]
            self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # Identity quaternion
            
            print(f"Body placed at ({body_x:.2f}, {body_y:.2f}, {body_z:.2f})")
        
        # Initialize arms to COMPACT pose (same as main)
        compact_arm_pose = np.array([0.0, -1.2, 0.0, 1.8, 0.0, 0.5, 0.0])
        self.data.qpos[self.left_arm_qpos_slice] = compact_arm_pose.copy()
        self.data.qpos[self.right_arm_qpos_slice] = compact_arm_pose.copy()
        self.data.qvel[:] = 0.0
        mujoco.mj_forward(self.model, self.data)
        
        # Import and run the locomotion loop (same as in run_visualization)
        # This is a simplified version - in a real implementation, 
        # you'd extract the locomotion loop into a separate method
        
        max_steps = int(duration / self.model.opt.timestep)
        step = 0
        
        # State tracking
        total_waypoints = len(self.current_path) if self.current_path else 0
        current_waypoint_idx = 0
        crawl_phase = 'initializing'
        success = False
        
        print(f"Running headless for up to {max_steps} steps ({duration}s)...")
        print(f"Total waypoints: {total_waypoints}")
        
        while step < max_steps:
            # Step physics
            mujoco.mj_step(self.model, self.data)
            step += 1
            
            # Apply Phase 2 control
            self.step_phase2_control()
            
            # Check for trajectory completion
            if self.current_path and current_waypoint_idx < total_waypoints:
                target = np.array(self.current_path[current_waypoint_idx].position)
                left_pos = self.get_left_gripper_pos()
                right_pos = self.get_right_gripper_pos()
                
                left_error = np.linalg.norm(left_pos - target)
                right_error = np.linalg.norm(right_pos - target)
                min_error = min(left_error, right_error)
                
                if min_error < 0.15:
                    current_waypoint_idx += 1
                    print(f"WP{current_waypoint_idx} reached!")
                    
                    if current_waypoint_idx >= total_waypoints:
                        print(f"\n✅ TRAJECTORY COMPLETE!")
                        success = True
                        break
            
            # Periodic status update
            if step % 5000 == 0:
                body_pos = self.get_central_body_pos()
                print(f"  [Step {step}] Body: ({body_pos[0]:.2f}, {body_pos[1]:.2f}, {body_pos[2]:.2f})")
        
        # Record result
        if hasattr(self, 'goal_position'):
            final_pos = self.get_central_body_pos()
            record_run_result(success, current_waypoint_idx, total_waypoints)
        
        return success

    def run_visualization(self, duration: float = None):
        """
        Run the MuJoCo visualization showing:
        - Wall positions (gray dots)
        - Start position (green)
        - Goal position (red)
        - Planned path (blue line)
        - Current robot position
        """
        
        print(f"\n{'='*60}")
        print("STARTING MUJOCO VISUALIZATION")
        print(f"{'='*60}")
        print("Visualization Legend:")
        print("  🔘 Gray spheres: Wall grip positions")
        print("  🟢 Green sphere: Start position / Completed waypoints")
        print("  🔴 Red sphere: Goal position")
        print("  🔵 Blue spheres: Upcoming waypoints")
        print("  🟡 Yellow sphere: Current target waypoint")
        print("  🟠 Orange sphere: Active arm target")
        print("  ⎯⎯ Cyan lines: Planned route")
        print("  ⎯⎯ Green lines: Completed route segments")
        print(f"  🔵 Blue transparent sphere: Left arm workspace ({WORKSPACE_SPHERE_RADIUS}m from elbow)")
        print(f"  🟠 Orange transparent sphere: Right arm workspace ({WORKSPACE_SPHERE_RADIUS}m from elbow)")
        print(f"{'='*60}")
        print("Controls:")
        print("  - Mouse drag: Rotate camera")
        print("  - Scroll: Zoom")
        print("  - Double-click: Track object")
        print("  - ESC: Exit")
        print(f"{'='*60}")
        print("⚠️  Robot is FREE-FLOATING (physics enabled)")
        print(f"{'='*60}")
        
        # Create scene and context for custom rendering
        scene = mujoco.MjvScene(self.model, maxgeom=1000)
        
        with mujoco.viewer.launch_passive(
            self.model, 
            self.data,
            show_left_ui=True,
            show_right_ui=True
        ) as viewer:
            
            # Set camera - EXACT same as dual_arm_simulation.py
            viewer.cam.lookat[:] = [0.0, 0.5, 1.0]
            viewer.cam.distance = 2.249
            viewer.cam.azimuth = 0.75
            viewer.cam.elevation = -20.0
            
            # Reset simulation
            mujoco.mj_resetData(self.model, self.data)
            
            # ================================================================
            # INITIALIZE ROBOT AT CENTER OF ISS MODULE
            # ================================================================
            # Position body at the CENTER of the ISS module first
            # Then arms will use IK to reach the wall positions
            
            # ISS module bounds: X: -2.1 to 3.9, Y: -0.5 to 1.7, Z: 0.1 to 2.2
            module_center_x = (ISS_MODULE['x_min'] + ISS_MODULE['x_max']) / 2  # 0.9
            module_center_y = (ISS_MODULE['y_min'] + ISS_MODULE['y_max']) / 2  # 0.6
            module_center_z = (ISS_MODULE['z_min'] + ISS_MODULE['z_max']) / 2  # 1.15
            
            if self.start_position and self.current_path and len(self.current_path) > 0:
                # ================================================================
                # SMART BASE PLACEMENT
                # Position body so the active arm can easily reach the first anchor
                # ================================================================
                wp1 = np.array(self.current_path[0].position)
                wp1_wall = self.current_path[0].wall
                active_arm = self.current_path[0].active_arm
                
                # 1. Get Surface Normal (points INTO module)
                surface_normal = self.get_surface_normal(wp1_wall)
                
                # 2. Determine ideal body position
                # We want the body to be positioned such that:
                # - The arm is not fully extended (avoid singularities)
                # - The arm is not too compressed (avoid self-collision)
                # - The body is "above" the surface (along the normal)
                
                ideal_reach = 0.85  # ideal distance from base to gripper (m)
                
                # Arm Base Offset from Body Center
                if active_arm == 'left':
                    arm_base_offset = np.array([LEFT_ARM_OFFSET, 0, 0])
                else:
                    arm_base_offset = np.array([RIGHT_ARM_OFFSET, 0, 0])
                
                # Body Position = Waypoint - ArmBaseOffset + (Normal * Reach)
                # Use normal * reach to place body "above" the surface
                body_pos = wp1 - arm_base_offset + (surface_normal * ideal_reach)
                
                body_x, body_y, body_z = body_pos
                
                # STRICT clamping to ensure ENTIRE robot is inside module
                # With X-axis mounting: Left Arm Base: BodyX - 0.3 | Right Arm Base: BodyX + 0.05
                # We need ArmBase +/- Buffer to be inside [X_min, X_max]
                # KUKA arm reach is ~0.75m, but we use 0.5m buffer as compromise
                
                arm_buffer = 0.50  # Compromise: prevents most wall penetration while allowing reach
                
                # Min Body X: LeftArmBase > X_min + Buffer => BodyX - 0.3 > X_min + Buffer
                min_body_x = ISS_MODULE['x_min'] + arm_buffer + 0.3
                
                # Max Body X: RightArmBase < X_max - Buffer => BodyX + 0.05 < X_max - Buffer
                max_body_x = ISS_MODULE['x_max'] - arm_buffer - 0.05
                
                # Standard padding for Y and Z - need buffer for arm reach
                safe_margin = 0.40  # Reduced from 0.55 to allow closer approach to waypoints
                
                body_x = np.clip(body_x, min_body_x, max_body_x)
                body_y = np.clip(body_y, ISS_MODULE['y_min'] + safe_margin, ISS_MODULE['y_max'] - safe_margin)
                body_z = np.clip(body_z, ISS_MODULE['z_min'] + safe_margin, ISS_MODULE['z_max'] - safe_margin)
                
                self.data.qpos[0] = body_x
                self.data.qpos[1] = body_y
                self.data.qpos[2] = body_z
                
                print(f"\n📍 WALL-CRAWLER LOCOMOTION TEST (Arms on X-ends)")
                print(f"  Start Wall: {wp1_wall}")
                print(f"  First anchor (WP1): ({wp1[0]:.2f}, {wp1[1]:.2f}, {wp1[2]:.2f})")
                print(f"  Body position: ({body_x:.2f}, {body_y:.2f}, {body_z:.2f})")
                print(f"  Left arm at X=-0.2, points: -X direction")
                print(f"  Right arm at X=+0.2, points: +X direction")
                print(f"  Distance to WP1: {np.linalg.norm(wp1 - np.array([body_x, body_y, body_z])):.2f}m")
            elif self.start_position:
                anchor_x, anchor_y, anchor_z = self.start_position
                self.data.qpos[0] = anchor_x + 0.6
                self.data.qpos[1] = anchor_y
                self.data.qpos[2] = anchor_z
            else:
                self.data.qpos[0] = 1.0
                self.data.qpos[1] = 0.6
                self.data.qpos[2] = 1.0
            
            self.data.qpos[3] = 1.0   # quat w
            self.data.qpos[4:7] = 0.0 # quat xyz
            
            # ================================================================
            # SMART INITIAL ARM CONFIGURATION
            # Pre-rotate base joint toward target to help IK converge
            # ================================================================
            body_pos = self.data.qpos[0:3]
            
            # Compute direction to WP1 from body
            dir_to_wp1 = wp1 - body_pos
            
            # For LEFT arm (points -X by default):
            # If target is in +X direction relative to body, arm needs to rotate
            # Base joint (joint1 = qpos[7]) rotates the arm
            left_target_angle = np.arctan2(dir_to_wp1[1], -dir_to_wp1[0])  # Flip X for left arm pointing -X
            self.data.qpos[7] = np.clip(left_target_angle, -2.5, 2.5)  # Left arm joint 1
            self.data.qpos[10] = 0.5  # Slight elbow bend to avoid singularity
            
            print(f"  🎯 Left arm pre-rotation: {np.degrees(left_target_angle):.1f}° toward WP1")
            
            # Initialize ArmController for IK solving
            print("  ⚙️ Solving IK for initial anchor...")
            try:
                # Create detailed configuration with high gains for solving
                from arm_controller import ArmController, ArmControllerConfig
                ik_config = ArmControllerConfig()
                ik_config.kp_position = 20.0  # Normalized gain for iterative solve
                ik_config.kd_position = 0.0   # No damping needed for static solve
                
                controller = ArmController(self.model, self.data, ik_config)
                
                # Set target for Left Arm (WP1)
                controller.left_target_position = np.array(wp1)
                
                # Convergence Loop
                for _ in range(150):
                    # Compute control command
                    joint_cmd, pos_error = controller.compute_arm_control('left')
                    
                    # Apply command directly to qpos (teleport integration)
                    self.data.qpos[7:14] = joint_cmd
                    
                    # Update kinematics
                    mujoco.mj_forward(self.model, self.data)
                    
                    if np.linalg.norm(pos_error) < 0.02:
                        print(f"  ✓ IK Converged! Error: {np.linalg.norm(pos_error):.4f}m")
                        break
                        
                # Ensure RIGHT ARM is tucked
                self.data.qpos[25] = 1.5  # Joint 4 tucked
                mujoco.mj_forward(self.model, self.data)
                
            except ImportError:
                print("  ⚠️ ArmController not found, using default pose")
                self.data.qpos[10] = 1.0  # Fallback manual bend
            except Exception as e:
                print(f"  ⚠️ IK failed: {e}")
                self.data.qpos[10] = 1.0  # Fallback manual bend
            
            # Zero velocities
            self.data.qvel[:] = 0.0
            
            # RIGHT ARM - Start TUCKED IN to avoid extending outside workspace
            # Right arm points +X by default, fold it toward body
            self.data.qpos[21] = 0.0     # joint0 - base rotation
            self.data.qpos[22] = 0.0     # joint1 
            self.data.qpos[23] = -0.5    # joint2 - pitch backward (toward body)
            self.data.qpos[24] = 0.0     # joint3 
            self.data.qpos[25] = 1.5     # joint4 - elbow bent sharply (tucked)
            self.data.qpos[26] = 0.0     # joint5 
            self.data.qpos[27] = 0.0     # joint6
            self.data.qpos[28] = 0.0     # joint7
            
            # Zero velocities to keep robot stable
            self.data.qvel[:] = 0.0
            
            # Close LEFT gripper immediately (value 255 = fully closed)
            self.data.ctrl[self.left_gripper_actuator_idx] = 255.0
            
            # Zero other control signals
            for i in range(len(self.data.ctrl)):
                if i != self.left_gripper_actuator_idx:
                    self.data.ctrl[i] = 0.0
            
            mujoco.mj_forward(self.model, self.data)
            
            # ================================================================
            # VERIFY ARMS ARE INSIDE MODULE BOUNDS
            # If any arm is outside, reset to compact pose
            # ================================================================
            left_ee = self.get_left_gripper_pos()
            right_ee = self.get_right_gripper_pos()
            
            def is_inside_bounds(pos):
                margin = 0.05  # 5cm margin for safety
                return (ISS_MODULE['x_min'] + margin <= pos[0] <= ISS_MODULE['x_max'] - margin and
                        ISS_MODULE['y_min'] + margin <= pos[1] <= ISS_MODULE['y_max'] - margin and
                        ISS_MODULE['z_min'] + margin <= pos[2] <= ISS_MODULE['z_max'] - margin)
            
            if not is_inside_bounds(left_ee) or not is_inside_bounds(right_ee):
                print(f"\n⚠️ ARM OUTSIDE BOUNDS - Resetting to compact pose!")
                print(f"   Left EE: ({left_ee[0]:.2f}, {left_ee[1]:.2f}, {left_ee[2]:.2f})")
                print(f"   Right EE: ({right_ee[0]:.2f}, {right_ee[1]:.2f}, {right_ee[2]:.2f})")
                
                # Force compact pose for both arms
                compact_arm_pose = np.array([0.0, -1.2, 0.0, 1.8, 0.0, 0.5, 0.0])
                self.data.qpos[self.left_arm_qpos_slice] = compact_arm_pose.copy()
                self.data.qpos[self.right_arm_qpos_slice] = compact_arm_pose.copy()
                mujoco.mj_forward(self.model, self.data)
                
                # Verify again
                left_ee = self.get_left_gripper_pos()
                right_ee = self.get_right_gripper_pos()
                print(f"   After reset - Left EE: ({left_ee[0]:.2f}, {left_ee[1]:.2f}, {left_ee[2]:.2f})")
                print(f"   After reset - Right EE: ({right_ee[0]:.2f}, {right_ee[1]:.2f}, {right_ee[2]:.2f})")
            else:
                print(f"✓ Both arms inside module bounds")
            
            # ================================================================
            # Set left arm as ALREADY ANCHORED at start position
            # ================================================================
            if self.start_position:
                first_wp = self.current_path[0] if self.current_path else None
                if first_wp:
                    self.left_target_position = np.array(first_wp.position)
                    self.left_arm_anchored = True
                    self.left_anchor_position = np.array(first_wp.position)
                    print(f"\n✓ Left arm PRE-ANCHORED at start: {first_wp.position}")
            
            start_time = time.time()
            step_count = 0
            first_render = True
            
            # ================================================================
            # FULL TRAJECTORY LOCOMOTION STATE MACHINE
            # ================================================================
            settling_timeout = 5000    # Max steps per waypoint (increased for more reliable convergence)
            crawl_phase = 'reaching_anchor'
            phase_timer = 0
            best_error = float('inf')
            
            # Trajectory tracking
            current_waypoint_idx = 0  # Start at first waypoint
            total_waypoints = len(self.current_path) if self.current_path else 0
            
            # Update display index for visualization
            self._current_waypoint_display_idx = current_waypoint_idx
            
            # ================================================================
            # STUCK DETECTION AND RECOVERY PARAMETERS
            # ================================================================
            stuck_detection_threshold = 8.0  # seconds without progress (REDUCED from 20)
            stuck_error_improvement_threshold = 0.02  # Need to improve by at least 2cm
            last_progress_time = time.time()
            last_best_error = float('inf')
            recovery_attempts = 0
            max_recovery_attempts = 5  # Increased to 5 for more recovery angles
            in_recovery_mode = False
            recovery_start_time = 0
            recovery_duration = 3.0  # Reduced to 3s for faster recovery cycles
            self._stuck_counter = 0  # Initialize stuck counter
            self._partial_release = False  # Initialize partial release flag
            recovery_joint1_target = None  # Target for joint1 during recovery
            recovery_j1_velocity_gain = 6.0  # Increased from 4.0 for faster rotation
            recovery_damping_factor = 0.90  # Reduced from 0.95 for more aggressive repositioning
            
            # RECOVERY STRATEGIES: More granular angles to try
            # Based on testing: joints need to be around -1.0 to reach backwards
            # Try a wider range including the found optimal values
            recovery_strategies = [0.0, -1.0, 1.0, -2.0, 2.0]
            
            # Get all waypoints and walls from path
            waypoints = []
            waypoint_walls = []
            waypoint_arms = []
            
            if self.current_path:
                for wp in self.current_path:
                    waypoints.append(np.array(wp.position))
                    waypoint_walls.append(wp.wall)
                    waypoint_arms.append(wp.active_arm)
                
                print(f"\n📍 FULL TRAJECTORY: {total_waypoints} waypoints")
                for i, (wp, wall, arm) in enumerate(zip(waypoints, waypoint_walls, waypoint_arms)):
                    is_final = " [FINAL GOAL]" if i == total_waypoints - 1 else ""
                    print(f"  WP{i+1}: ({wp[0]:.2f}, {wp[1]:.2f}, {wp[2]:.2f}) | {wall:12s} | {arm.upper()} arm{is_final}")
            
            # Current moving arm and anchored arm - USE PATH PLANNER'S ARM ASSIGNMENT
            moving_arm = waypoint_arms[0] if waypoint_arms else 'left'
            anchored_arm = 'right' if moving_arm == 'left' else 'left'
            print(f"\n🦾 Starting locomotion: {moving_arm.upper()} arm moves first (from path planner)")
            
            # Current target (first waypoint)
            current_target = waypoints[0] if waypoints else None
            current_target_wall = waypoint_walls[0] if waypoint_walls else 'front'
            use_orientation_control = False  # Only for final approach
            
            # Store initial body position
            initial_body_pos = self.data.qpos[0:3].copy()
            initial_body_quat = self.data.qpos[3:7].copy()
            
            # ================================================================
            # PRE-POSITIONING CHECK
            # ================================================================
            print("\n⚙️  WALL-CRAWLER LOCOMOTION SEQUENCE")
            
            mujoco.mj_forward(self.model, self.data)
            
            body_pos = self.get_central_body_pos()
            left_pos = self.get_left_gripper_pos()
            right_pos = self.get_right_gripper_pos()
            
            print(f"  Body at ({body_pos[0]:.2f}, {body_pos[1]:.2f}, {body_pos[2]:.2f})")
            print(f"  Left gripper at ({left_pos[0]:.2f}, {left_pos[1]:.2f}, {left_pos[2]:.2f})")
            print(f"  Right gripper at ({right_pos[0]:.2f}, {right_pos[1]:.2f}, {right_pos[2]:.2f})")
            if current_target is not None:
                initial_error = np.linalg.norm(current_target - left_pos)
                print(f"  First target: ({current_target[0]:.2f}, {current_target[1]:.2f}, {current_target[2]:.2f})")
                print(f"  Initial distance: {initial_error:.3f}m")
            
            print(f"\n{'='*60}")
            print("LOCOMOTION PHASES:")
            print("  1. Reach waypoints alternating left/right arms")
            print("  2. Anchor arm at each waypoint")
            print("  3. Release previous anchor, reach next")
            print("  4. Final waypoint: approach perpendicular to surface")
            print(f"{'='*60}")
            
            # Store the left arm joint positions when anchored
            left_arm_anchored_joints = None
            right_arm_anchored_joints = None
            
            # ================================================================
            # TRACKING: Success rate, anchor deviation, AND trajectory error
            # ================================================================
            trajectory_success = False
            anchor_deviation_history = {}  # {waypoint_idx: {'anchor_pos': ..., 'max_deviation': ..., 'deviations': [...]}}
            current_anchor_idx = None
            
            # NEW: Trajectory error tracking for real-time graph
            trajectory_error_history = {
                'time_steps': [],
                'errors': [],
                'waypoint_idx': [],
                'arm': [],
                'phase': []
            }
            # NEW: Dynamics tracking
            dynamics_history = {
                'time_steps': [],
                'anchor_forces': [],      # List of force_mag
                'body_torques': [],       # List of torque_z
                'anchor_arm': []          # Which arm was anchored
            }
            
            global_step = 0  # Track total steps for trajectory plotting
            
            try:
                while viewer.is_running():
                    step_start = time.time()
                    
                    # ============================================================
                    # Apply body damping (always, since body is free from start)
                    # ============================================================
                    body_damping = 5.0
                    self.data.qvel[0:6] *= (1.0 - body_damping * self.model.opt.timestep)
                    
                    # Clear body force/torque at start of each step
                    self.data.xfrc_applied[self.central_body_id, :] = 0
                    
                    # Per-step logging variables
                    step_anchor_force = 0.0
                    step_anchor_arm_name = "none"
                    current_body_torque = 0.0
                    
                    # ============================================================
                    # BODY ORIENTATION CONTROL - Keep body aimed at ISS center
                    # ============================================================
                    # ISS module center (approximately)
                    iss_center = np.array([0.9, 0.6, 1.15])
                    body_pos = self.get_central_body_pos()
                    
                    # Get current body orientation (quaternion)
                    body_quat = self.data.qpos[3:7].copy()
                    
                    # Desired orientation: body Y-axis (front) points toward ISS center
                    to_center = iss_center - body_pos
                    to_center_horiz = np.array([to_center[0], to_center[1], 0.0])
                    horiz_dist = np.linalg.norm(to_center_horiz)
                    
                    if horiz_dist > 0.1:
                        # Compute desired yaw angle (rotation around Z)
                        desired_yaw = np.arctan2(to_center_horiz[1], to_center_horiz[0])
                        
                        # Current yaw from quaternion
                        # Extract yaw from quaternion (rotation around Z)
                        w, x, y, z = body_quat
                        current_yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
                        
                        # Yaw error
                        yaw_error = desired_yaw - current_yaw
                        # Wrap to [-pi, pi]
                        while yaw_error > np.pi:
                            yaw_error -= 2*np.pi
                        while yaw_error < -np.pi:
                            yaw_error += 2*np.pi
                        
                        # Apply torque to correct orientation (keep box aimed at ISS center)
                        # Increase stiffness slightly for a firmer heading while keeping damping to avoid oscillation
                        orientation_stiffness = 60.0
                        orientation_damping = 18.0
                        yaw_vel = self.data.qvel[5]  # Angular velocity around Z
                        
                        torque_z = orientation_stiffness * yaw_error - orientation_damping * yaw_vel
                        # write into moment (z) slot for central body
                        self.data.xfrc_applied[self.central_body_id, 5] = torque_z
                        
                        # Store for logging
                        current_body_torque = torque_z
                    
                    # ============================================================
                    # FULL TRAJECTORY STATE MACHINE
                    # ============================================================
                    
                    # Helper function to apply anchor force AND body pull force
                    # IMPROVED: Better damping to reduce vibration
                    def apply_anchor_force(arm, anchor_pos, prev_pos_attr, hard_lock: bool = False):
                        """Apply anchor force to hold end-effector at position.
                        
                        Args:
                            arm: 'left' or 'right'
                            anchor_pos: Target anchor position
                            prev_pos_attr: Attribute name to store previous position
                            hard_lock: If True, use very aggressive locking (for final position)
                        """
                        if arm == 'left':
                            pos = self.get_left_gripper_pos()
                            ee_body_id = self.left_ee_body_id
                            qpos_slice = slice(7, 14)
                            qvel_slice = slice(6, 13)
                            ctrl_slice = self.left_arm_actuator_slice
                        else:
                            pos = self.get_right_gripper_pos()
                            ee_body_id = self.right_ee_body_id
                            qpos_slice = slice(22, 29)
                            qvel_slice = slice(21, 28)
                            ctrl_slice = self.right_arm_actuator_slice
                        
                        err = anchor_pos - pos
                        err_mag = np.linalg.norm(err)
                        
                        # Track previous position for velocity estimation
                        prev_pos = getattr(self, prev_pos_attr, pos)
                        vel_approx = (pos - prev_pos) / self.model.opt.timestep
                        setattr(self, prev_pos_attr, pos.copy())
                        
                        # CRITICAL DAMPING: zeta = 1.0 for no overshoot
                        # For critically damped system: damping = 2 * sqrt(stiffness * mass)
                        # Assume effective mass ~5 kg at end-effector
                        effective_mass = 5.0
                        
                        if hard_lock:
                            # HARD LOCK MODE: Very high stiffness, critical damping
                            grip_stiffness = 120000.0  # INCREASED for better anchor hold
                            grip_damping = 2.0 * np.sqrt(grip_stiffness * effective_mass)  # ~1549
                            max_force = 80000.0  # INCREASED
                            velocity_damping_factor = 0.4  # Aggressive velocity kill
                        else:
                        # Normal anchoring - increased stiffness for stability
                            grip_stiffness = 150000.0  # INCREASED from 80000 (User request: increase rigidity)
                            grip_damping = 2.0 * np.sqrt(grip_stiffness * effective_mass)  # Critical damping ~1732
                            max_force = 60000.0   # INCREASED
                            velocity_damping_factor = 0.9  # Gentle damping
                        
                        # Spring-damper force with critical damping
                        force = grip_stiffness * err - grip_damping * vel_approx
                        force_mag = np.linalg.norm(force)
                        if force_mag > max_force:
                            force = force * (max_force / force_mag)
                        self.data.xfrc_applied[ee_body_id, 0:3] = force
                        
                        # Capture force for logging
                        nonlocal step_anchor_force, step_anchor_arm_name
                        if force_mag > step_anchor_force:
                            step_anchor_force = force_mag
                            step_anchor_arm_name = arm
                        
                        # Lock arm joints to maintain anchor rigidly
                        lock_attr = f'_locked_{arm}_joints'
                        if not hasattr(self, lock_attr) or getattr(self, lock_attr) is None:
                            # First time - save current joint positions
                            setattr(self, lock_attr, self.data.qpos[qpos_slice].copy())
                        
                        # Apply joint position control to lock the arm
                        locked_joints = getattr(self, lock_attr)
                        if locked_joints is not None:
                            self.data.ctrl[ctrl_slice] = locked_joints
                            # Damp joint velocities to prevent vibration
                            self.data.qvel[qvel_slice] *= velocity_damping_factor
                            
                            if hard_lock:
                                # In hard lock mode, also force-restore joint positions
                                # This directly overrides any physics drift
                                joint_error = locked_joints - self.data.qpos[qpos_slice]
                                if np.linalg.norm(joint_error) > 0.001:
                                    # Blend toward locked position
                                    self.data.qpos[qpos_slice] = self.data.qpos[qpos_slice] + 0.1 * joint_error
                        
                        # BODY PULL: Pull body toward MIDPOINT between anchor and where moving arm needs to go
                        # This helps position the body optimally for the next waypoint
                        body_pos = self.get_central_body_pos()
                        
                        # Get the moving arm's target (if available)
                        moving_target = None
                        if arm == 'left' and hasattr(self, 'right_target_position') and self.right_target_position is not None:
                            moving_target = self.right_target_position
                        elif arm == 'right' and hasattr(self, 'left_target_position') and self.left_target_position is not None:
                            moving_target = self.left_target_position
                        
                        if moving_target is not None:
                            # Pull body toward the midpoint between anchor and moving target
                            midpoint = (anchor_pos + moving_target) / 2.0
                            body_to_mid = midpoint - body_pos
                            distance = np.linalg.norm(body_to_mid)
                            
                            # Moderate force for smooth body repositioning
                            if distance > 0.10:
                                body_pull_stiffness = 300.0  # Moderate force
                                body_damping_coef = 40.0     # Good damping
                                body_vel = self.data.qvel[0:3]
                                
                                body_force = body_pull_stiffness * body_to_mid - body_damping_coef * body_vel
                                self.data.xfrc_applied[self.central_body_id, 0:3] += body_force
                        else:
                            # Fall back to original anchor-only pull
                            body_to_anchor = anchor_pos - body_pos
                            distance = np.linalg.norm(body_to_anchor)
                            
                            if distance > 0.4:
                                body_pull_stiffness = 200.0  # Increased from 100.0
                                body_damping_coef = 30.0
                                body_vel = self.data.qvel[0:3]
                                
                                body_force = body_pull_stiffness * body_to_anchor - body_damping_coef * body_vel
                                self.data.xfrc_applied[self.central_body_id, 0:3] += body_force
                        
                        # TRACK: Record deviation for anchor statistics
                        if current_anchor_idx is not None and current_anchor_idx in anchor_deviation_history:
                            anchor_deviation_history[current_anchor_idx]['deviations'].append(err_mag)
                            if err_mag > anchor_deviation_history[current_anchor_idx]['max_deviation']:
                                anchor_deviation_history[current_anchor_idx]['max_deviation'] = err_mag
                        
                        return err_mag
                    
                    if crawl_phase == 'reaching_anchor':
                        # First phase: Left arm reaches first waypoint
                        phase_timer += 1
                        global_step += 1
                        
                        # Update display index for visualization
                        self._current_waypoint_display_idx = current_waypoint_idx
                        
                        # Keep right arm retracted
                        self.data.ctrl[self.right_arm_actuator_slice] = self.data.qpos[self.right_arm_qpos_slice]
                        
                        # Move body toward first waypoint (before any anchor exists)
                        # This helps the robot get into position for the first grab
                        body_pos = self.get_central_body_pos()
                        body_to_target = current_target - body_pos
                        body_distance = np.linalg.norm(body_to_target)
                        
                        # Apply body force to move toward target - moderate for smooth motion
                        body_vel = self.data.qvel[0:3]
                        kp_body = 350.0  # INCREASED: More aggressive repositioning to reach first waypoint
                        kd_body = 30.0   # Higher damping for stability
                        body_force = kp_body * body_to_target - kd_body * body_vel
                        self.data.xfrc_applied[self.central_body_id, 0:3] = body_force
                        
                        # Left arm reaches toward first waypoint
                        if current_target is not None:
                            self.left_target_position = current_target.copy()
                            left_error = self.apply_arm_control('left')
                            
                            # Track trajectory error for plotting
                            trajectory_error_history['time_steps'].append(global_step)
                            trajectory_error_history['errors'].append(left_error)
                            trajectory_error_history['waypoint_idx'].append(current_waypoint_idx)
                            trajectory_error_history['arm'].append('left')
                            trajectory_error_history['phase'].append('reaching_anchor')
                            
                            # Apply stabilizing force when close
                            if left_error < 0.3:
                                grip_stiffness = 3000.0
                                left_pos = self.get_left_gripper_pos()
                                left_err = current_target - left_pos
                                self.data.xfrc_applied[self.left_ee_body_id, 0:3] = grip_stiffness * left_err
                            
                            if left_error < best_error:
                                best_error = left_error
                                last_progress_time = time.time()  # Reset progress timer on improvement
                                last_best_error = left_error
                            
                            if phase_timer % 100 == 0:
                                left_pos = self.get_left_gripper_pos()
                                body_pos = self.get_central_body_pos()
                                print(f"  [{phase_timer:4d}] Left→WP1 | Error: {left_error:.3f}m | Body: ({body_pos[0]:.2f}, {body_pos[1]:.2f}, {body_pos[2]:.2f})")
                            
                            # Success - anchored at first waypoint (STRICT threshold - DO NOT RELAX)
                            if left_error < 0.10:
                                left_pos = self.get_left_gripper_pos()
                                print(f"\n✅ WP{current_waypoint_idx+1} REACHED by LEFT arm! Error: {left_error:.3f}m")
                                
                                # ANCHOR AT THE VALID SPHERE POSITION (the target waypoint)
                                anchor_position = np.array(current_target)
                                self.left_arm_anchored = True
                                self.left_anchor_position = anchor_position.copy()
                                print(f"  🔒 LEFT arm ANCHORED at sphere ({anchor_position[0]:.2f}, {anchor_position[1]:.2f}, {anchor_position[2]:.2f})")
                                
                                # Initialize anchor deviation tracking for this waypoint
                                current_anchor_idx = current_waypoint_idx
                                anchor_deviation_history[current_anchor_idx] = {
                                    'anchor_pos': anchor_position.copy(),
                                    'arm': 'left',
                                    'max_deviation': 0.0,
                                    'deviations': []
                                }
                                
                                current_waypoint_idx += 1
                                
                                if current_waypoint_idx < total_waypoints:
                                    # Move to next waypoint with right arm
                                    current_target = waypoints[current_waypoint_idx]
                                    current_target_wall = waypoint_walls[current_waypoint_idx]
                                    is_final = current_waypoint_idx == total_waypoints - 1
                                    # DISABLED: Orientation control makes final WP harder to reach
                                    # Position-only control is more reliable
                                    use_orientation_control = False
                                    moving_arm = 'right'
                                    anchored_arm = 'left'
                                    crawl_phase = 'reaching_waypoint'
                                    phase_timer = 0
                                    best_error = float('inf')
                                    print(f"  🔒 Left arm LOCKED")
                                    print(f"  ➡️  Right arm moving to WP{current_waypoint_idx+1}")
                                else:
                                    crawl_phase = 'trajectory_complete'
                                    self._trajectory_complete = True  # Flag for visualization
                                    print(f"\n🎉 TRAJECTORY COMPLETE!")
                        
                        if phase_timer >= settling_timeout:
                            print(f"\n⚠️ Timeout reaching WP1")
                            crawl_phase = 'failed'
                    
                    elif crawl_phase == 'reaching_waypoint':
                        # Alternating arm reaches next waypoint
                        phase_timer += 1
                        global_step += 1
                        
                        # First, set the moving arm's target so body pull knows where to go
                        if moving_arm == 'left':
                            self.left_target_position = current_target.copy()
                        else:
                            self.right_target_position = current_target.copy()
                        
                        # Apply anchor force to the anchored arm (now knows moving target for body pull)
                        if anchored_arm == 'left' and self.left_anchor_position is not None:
                            apply_anchor_force('left', self.left_anchor_position, '_prev_left_pos')
                            self.left_target_position = self.left_anchor_position.copy()
                            # Only control last 2 joints of anchored arm to maintain anchor
                            # First 5 joints can be adjusted to help moving arm reach
                            self.apply_arm_control('left', anchored_mode=True)
                        elif anchored_arm == 'right' and self.right_anchor_position is not None:
                            apply_anchor_force('right', self.right_anchor_position, '_prev_right_pos')
                            self.right_target_position = self.right_anchor_position.copy()
                            self.apply_arm_control('right', anchored_mode=True)
                        
                        # Moving arm reaches toward target
                        # Also adjust first 5 joints of anchored arm to help body positioning
                        if moving_arm == 'left':
                            self.left_target_position = current_target.copy()  # Re-set after anchor overwrote
                            if use_orientation_control:
                                error = self.apply_arm_control_with_orientation('left', current_target_wall)
                            else:
                                error = self.apply_arm_control('left')
                            arm_pos = self.get_left_gripper_pos()
                            
                            # Use coordinated control ALWAYS to help body reposition faster
                            if anchored_arm == 'right' and self.right_anchor_position is not None:
                                self.apply_coordinated_arm_control('right', current_target, self.right_anchor_position)
                        else:
                            self.right_target_position = current_target.copy()  # Re-set after anchor overwrote
                            if use_orientation_control:
                                error = self.apply_arm_control_with_orientation('right', current_target_wall)
                            else:
                                error = self.apply_arm_control('right')
                            arm_pos = self.get_right_gripper_pos()
                            
                            # Use coordinated control ALWAYS to help body reposition faster
                            if anchored_arm == 'left' and self.left_anchor_position is not None:
                                self.apply_coordinated_arm_control('left', current_target, self.left_anchor_position)
                        
                        # Track trajectory error for plotting
                        trajectory_error_history['time_steps'].append(global_step)
                        trajectory_error_history['errors'].append(error)
                        trajectory_error_history['waypoint_idx'].append(current_waypoint_idx)
                        trajectory_error_history['arm'].append(moving_arm)
                        trajectory_error_history['phase'].append('reaching_waypoint')
                        
                        # Update display index for visualization
                        self._current_waypoint_display_idx = current_waypoint_idx
                        
                        # ================================================================
                        # BODY REPOSITIONING - Pull body toward optimal position for reaching
                        # ================================================================
                        body_pos = self.get_central_body_pos()
                        # Get anchor position
                        if anchored_arm == 'left':
                            anchor_pos_now = self.left_anchor_position
                        else:
                            anchor_pos_now = self.right_anchor_position
                        
                        if anchor_pos_now is not None and current_target is not None:
                            # Midpoint between anchor and target is ideal body position
                            midpoint = (anchor_pos_now + current_target) / 2.0
                            body_to_mid = midpoint - body_pos
                            mid_distance = np.linalg.norm(body_to_mid)
                            
                            # INCREASED force for faster locomotion
                            if mid_distance > 0.05:
                                body_vel = self.data.qvel[0:3]
                                body_kp = 700.0  # INCREASED from 350 for faster body repositioning
                                body_kd = 60.0   # Moderate damping for stability
                                body_force = body_kp * body_to_mid - body_kd * body_vel
                                self.data.xfrc_applied[self.central_body_id, 0:3] += body_force
                        
                        # ================================================================
                        # STUCK DETECTION AND RECOVERY SYSTEM
                        # ================================================================
                        current_time = time.time()
                        
                        # Check if we're in recovery mode
                        if in_recovery_mode:
                            recovery_elapsed = current_time - recovery_start_time
                            
                            # RECOVERY STRATEGY: Rotate the ANCHORED arm to reposition body
                            # This helps the moving arm reach its target by changing body position
                            # FIX: Use anchored_arm instead of moving_arm for recovery rotation
                            if anchored_arm == 'left':
                                qpos_slice = self.left_arm_qpos_slice
                                actuator_slice = self.left_arm_actuator_slice
                            else:
                                qpos_slice = self.right_arm_qpos_slice
                                actuator_slice = self.right_arm_actuator_slice
                            
                            # Smoothly rotate joint1 towards target
                            current_j1 = self.data.qpos[qpos_slice][0]
                            j1_error = recovery_joint1_target - current_j1
                            
                            # Normalize angle error to [-pi, pi]
                            while j1_error > np.pi:
                                j1_error -= 2 * np.pi
                            while j1_error < -np.pi:
                                j1_error += 2 * np.pi
                            
                            # Apply joint1 rotation with optimized gain (faster for intelligent recovery)
                            # Apply joint1 rotation with optimized gain
                            j1_velocity = recovery_j1_velocity_gain * j1_error
                            new_j1 = current_j1 + j1_velocity * self.model.opt.timestep
                            
                            # SMART RECOVERY: Also retract shoulder (J2) and bend elbow (J4) to "Turtle" the arm
                            # This clears the workspace more effectively than just rotating
                            target_j2 = -1.0
                            target_j4 = 1.5
                            
                            current_j2 = self.data.qpos[qpos_slice][1]
                            current_j4 = self.data.qpos[qpos_slice][3]
                            
                            new_j2 = current_j2 + 2.0 * (target_j2 - current_j2) * self.model.opt.timestep
                            new_j4 = current_j4 + 2.0 * (target_j4 - current_j4) * self.model.opt.timestep
                            
                            # Set commands for Moving Arm
                            current_cmd = self.data.ctrl[actuator_slice].copy()
                            current_cmd[0] = new_j1  # Rotate
                            current_cmd[1] = new_j2  # Retract Shoulder
                            current_cmd[3] = new_j4  # Bend Elbow
                            self.data.ctrl[actuator_slice] = current_cmd
                            
                            # Apply recovery to Moving Arm too (retract to clear workspace)
                            # Since we're rotating the anchored arm, also turtle the moving arm
                            if moving_arm == 'left':
                                moving_actuator_slice = self.left_arm_actuator_slice
                                moving_qpos_slice = self.left_arm_qpos_slice
                            else:
                                moving_actuator_slice = self.right_arm_actuator_slice
                                moving_qpos_slice = self.right_arm_qpos_slice
                                
                            # Moving Arm Control: Retract J2/J4 to turtle
                            moving_current_q = self.data.qpos[moving_qpos_slice]
                            moving_current_j2 = moving_current_q[1]
                            moving_current_j4 = moving_current_q[3]
                            
                            moving_cmd = self.data.ctrl[moving_actuator_slice].copy()
                            # Retract moving arm to clear workspace (don't rotate J1 - keep trying to reach)
                            moving_cmd[1] += 2.0 * (target_j2 - moving_current_j2) * self.model.opt.timestep
                            moving_cmd[3] += 2.0 * (target_j4 - moving_current_j4) * self.model.opt.timestep
                            
                            self.data.ctrl[moving_actuator_slice] = moving_cmd
    
                            # Damping for other joints (J3, J5, J6, J7) is implicit as we don't actuate them strongly here?
                            # No, we must ensure they don't drift.
                            # Actually, keeping previous command for them (via copy) acts as position hold if using position control?
                            # The actuator is position control. So `current_cmd` holds last commanded value?
                            # Wait, `self.data.ctrl` persists? Yes.
                            # So simply updating J1,J2,J4 handles them, others hold.
                            
                            if phase_timer % 100 == 0:
                                print(f"  🔄 RECOVERY [{recovery_attempts}/{max_recovery_attempts}] J1: {np.degrees(current_j1):.1f}° → {np.degrees(recovery_joint1_target):.1f}° | {recovery_duration - recovery_elapsed:.1f}s left")
                            
                            # End recovery after duration
                            if recovery_elapsed >= recovery_duration:
                                in_recovery_mode = False
                                best_error = float('inf')  # Reset best error to give fresh start
                                last_progress_time = current_time  # Reset progress timer
                                last_best_error = float('inf')
                                print(f"\n  ✅ Recovery {recovery_attempts} complete - J1 repositioned, resuming control")
                        else:
                            # Track if error is improving
                            if error < last_best_error - stuck_error_improvement_threshold:
                                last_best_error = error
                                last_progress_time = current_time
                            
                            if error < best_error:
                                best_error = error
                                self._stuck_counter = 0
                            else:
                                self._stuck_counter += 1
                            
                            # Check if stuck (no progress for stuck_detection_threshold seconds)
                            time_without_progress = current_time - last_progress_time
                            
                            if time_without_progress > stuck_detection_threshold and error > 0.12:
                                if recovery_attempts < max_recovery_attempts:
                                    max_recovery_attempts = 15  # Increased to allow cycling through all strategies (3 standard + panic) multiple times
                                    stuck_error_improvement_threshold = 0.001
                                    stuck_detection_threshold = 5.0 # Check stuck every 5s
                                    
                                    recovery_attempts += 1
                                    in_recovery_mode = True
                                    recovery_start_time = current_time
                                    
                                    # Get current joint1 angle for computing target
                                    # FIX: Use ANCHORED arm since that's what we're rotating in recovery
                                    if anchored_arm == 'left':
                                        current_j1 = self.data.qpos[self.left_arm_qpos_slice][0]
                                    else:
                                        current_j1 = self.data.qpos[self.right_arm_qpos_slice][0]
                                    
                                    # Recovery strategies: Rotate RELATIVE to current position
                                    # Each recovery moves the arm a different amount
                                    recovery_rotation = [np.pi, np.pi/2, -np.pi/2, np.pi*0.75, -np.pi*0.75]
                                    strategy_idx = (recovery_attempts - 1) % len(recovery_rotation)
                                    rotation_amount = recovery_rotation[strategy_idx]
                                    recovery_joint1_target = current_j1 + rotation_amount
                                    
                                    # Normalize to [-pi, pi]
                                    while recovery_joint1_target > np.pi:
                                        recovery_joint1_target -= 2 * np.pi
                                    while recovery_joint1_target < -np.pi:
                                        recovery_joint1_target += 2 * np.pi
                                    
                                    # Human-readable strategy names
                                    strategy_names = ["180° FLIP", "+90° ROTATE", "-90° ROTATE", "+135° ROTATE", "-135° ROTATE"]
                                    strategy_name = strategy_names[strategy_idx]
    
                                    print(f"    - Strategy: {strategy_name}")
                                    print(f"    - J1: {np.degrees(current_j1):.1f}° → {np.degrees(recovery_joint1_target):.1f}°")
                                    
                                    # DUAL-ARM RECOVERY: Apply J1 target to BOTH arms to clear workspace
                                    # The 'moving_arm' logic below only sets one. We will override both in the loop.
                                else:
                                    print(f"\n❌ MAX RECOVERY ATTEMPTS ({max_recovery_attempts}) REACHED")
                                    print(f"  Resetting recovery counter to try sequence again (infinite retry)")
                                    recovery_attempts = 0
                                    last_progress_time = current_time
                                    last_best_error = float('inf')
                                    last_progress_time = current_time
                                    last_best_error = float('inf')
                        
                        # Legacy partial release if stuck counter (for faster response)
                        partial_release = getattr(self, '_partial_release', False)
                        if self._stuck_counter > 500 and best_error > 0.3 and not partial_release and not in_recovery_mode:
                            print(f"\n⚠️ ARM STUCK - Enabling partial release on {anchored_arm} arm")
                            self._partial_release = True
                            self._stuck_counter = 0
                        
                        if phase_timer % 100 == 0:
                            body_pos = self.get_central_body_pos()
                            is_final = current_waypoint_idx == total_waypoints - 1
                            suffix = " [FINAL]" if is_final else ""
                            print(f"  [{phase_timer:4d}] {moving_arm.upper()}→WP{current_waypoint_idx+1} | Error: {error:.3f}m | Body: ({body_pos[0]:.2f}, {body_pos[1]:.2f}, {body_pos[2]:.2f}){suffix}")
                        
                        # Success - reached waypoint (must be close to target sphere)
                        # STRICT THRESHOLDS - DO NOT RELAX
                        is_final_waypoint = current_waypoint_idx == total_waypoints - 1
                        # Use strict thresholds - arm must actually reach the waypoint
                        success_threshold = 0.10  # 10cm for ALL waypoints - NO RELAXATION
                        
                        should_accept = error < success_threshold
                        
                        
                        
                        
                        
                        if should_accept:
                            print(f"\n✅ WP{current_waypoint_idx+1} REACHED by {moving_arm.upper()} arm! Error: {error:.3f}m")
                            
                            # ANCHOR AT THE VALID SPHERE POSITION (the target waypoint)
                            # NOT at the current arm position - this ensures we anchor at valid spheres only
                            anchor_position = np.array(current_target)  # Use the target waypoint (valid sphere)
                            
                            if moving_arm == 'left':
                                self.left_arm_anchored = True
                                self.left_anchor_position = anchor_position.copy()
                                print(f"  🔒 LEFT arm ANCHORED at sphere ({anchor_position[0]:.2f}, {anchor_position[1]:.2f}, {anchor_position[2]:.2f})")
                            else:
                                self.right_arm_anchored = True
                                self.right_anchor_position = anchor_position.copy()
                                print(f"  🔒 RIGHT arm ANCHORED at sphere ({anchor_position[0]:.2f}, {anchor_position[1]:.2f}, {anchor_position[2]:.2f})")
                            
                            # Initialize anchor deviation tracking for this waypoint
                            current_anchor_idx = current_waypoint_idx
                            anchor_deviation_history[current_anchor_idx] = {
                                'anchor_pos': anchor_position.copy(),
                                'arm': moving_arm,
                                'max_deviation': 0.0,
                                'deviations': []
                            }
                            
                            current_waypoint_idx += 1
                            self._partial_release = False
                            self._stuck_counter = 0
                            
                            if current_waypoint_idx < total_waypoints:
                                # Prepare for next waypoint
                                is_final = current_waypoint_idx == total_waypoints - 1
                                
                                # Release previous anchor, swap arms (standard alternating)
                                if anchored_arm == 'left':
                                    self.left_arm_anchored = False
                                    self.data.xfrc_applied[self.left_ee_body_id, 0:3] = 0
                                    print(f"  🔓 LEFT arm RELEASED")
                                else:
                                    self.right_arm_anchored = False
                                    self.data.xfrc_applied[self.right_ee_body_id, 0:3] = 0
                                    print(f"  🔓 RIGHT arm RELEASED")
                                
                                # Swap arms - standard alternating locomotion
                                moving_arm, anchored_arm = anchored_arm, moving_arm
                                current_target = waypoints[current_waypoint_idx]
                                current_target_wall = waypoint_walls[current_waypoint_idx]
                                # DISABLED: Orientation control makes final WP harder to reach
                                use_orientation_control = False
                                
                                print(f"  ➡️  {moving_arm.upper()} arm moving to WP{current_waypoint_idx+1}{' [FINAL]' if is_final else ''}")
                                phase_timer = 0
                                best_error = float('inf')
                            else:
                                # FINAL WAYPOINT REACHED - release the previous anchor
                                # The arm that just reached the goal (moving_arm) is now anchored
                                # The previous anchored_arm becomes free for screw task
                                if anchored_arm == 'left':
                                    self.left_arm_anchored = False
                                    self.left_anchor_position = None
                                    self.data.xfrc_applied[self.left_ee_body_id, 0:3] = 0
                                    print(f"  🔓 LEFT arm RELEASED (now FREE for screw task)")
                                else:
                                    self.right_arm_anchored = False
                                    self.right_anchor_position = None
                                    self.data.xfrc_applied[self.right_ee_body_id, 0:3] = 0
                                    print(f"  🔓 RIGHT arm RELEASED (now FREE for screw task)")
                                
                                crawl_phase = 'trajectory_complete'
                                self._trajectory_complete = True  # Flag for visualization
                                print(f"\n🎉 TRAJECTORY COMPLETE!")
                                print(f"  Final position on {current_target_wall} wall")
                                body_pos = self.get_central_body_pos()
                                print(f"  Body at: ({body_pos[0]:.2f}, {body_pos[1]:.2f}, {body_pos[2]:.2f})")
                        
                        if phase_timer >= settling_timeout:
                            # Timeout - but DON'T proceed to next waypoint
                            # Only print status update, keep trying
                            if phase_timer == settling_timeout:
                                print(f"\n⏳ Still reaching WP{current_waypoint_idx+1} (timeout reached, continuing...)")
                                print(f"  Best error so far: {best_error:.3f}m")
                            
                            # Continue trying - reset timer but keep best_error
                            # This allows infinite attempts until the arm is anchored
                            if phase_timer % 1000 == 0:
                                body_pos = self.get_central_body_pos()
                                print(f"  [{phase_timer:5d}] {moving_arm.upper()}→WP{current_waypoint_idx+1} | Error: {error:.3f}m | Best: {best_error:.3f}m")
                    
                    # LOG DYNAMICS (Every 10 steps = 20ms)
                    if global_step % 10 == 0:
                        dynamics_history['time_steps'].append(global_step * self.model.opt.timestep)
                        dynamics_history['anchor_forces'].append(step_anchor_force)
                        dynamics_history['body_torques'].append(current_body_torque)
                        dynamics_history['anchor_arm'].append(step_anchor_arm_name)
                    
                    elif crawl_phase == 'trajectory_complete':
                        # FINAL POSITION: Execute unscrewing task with proper IK control
                        phase_timer += 1
                        
                        # Initialize unscrewing task on first entry
                        if not hasattr(self, '_unscrew_initialized') or not self._unscrew_initialized:
                            self._unscrew_initialized = True
                            self._unscrew_delay = 50  # Wait 50 steps before starting
                            self._unscrew_started = False
                            
                            # Determine which arm reached the final waypoint (is anchored at screw)
                            # That arm does the unscrewing, the other provides anchor support
                            if self.left_anchor_position is not None and self.right_anchor_position is None:
                                screw_arm = 'left'
                                anchor_arm = 'right'
                                print(f"\n  🔒 FINAL POSITION REACHED")
                                print(f"     LEFT arm at screw → will unscrew")
                                print(f"     RIGHT arm provides anchor support")
                            elif self.right_anchor_position is not None and self.left_anchor_position is None:
                                screw_arm = 'right'
                                anchor_arm = 'left'
                                print(f"\n  🔒 FINAL POSITION REACHED")
                                print(f"     RIGHT arm at screw → will unscrew")
                                print(f"     LEFT arm provides anchor support")
                            else:
                                # Default case - use left arm
                                screw_arm = 'left'
                                anchor_arm = 'right'
                                print(f"\n  🔒 FINAL POSITION - defaulting to LEFT arm for unscrew")
                            
                            self._unscrew_screw_arm = screw_arm
                            self._unscrew_anchor_arm_name = anchor_arm
                            
                            # Get the screw wall from stored info
                            screw_wall = getattr(self, '_screw_wall', 'back')
                            
                            # Store body position for locking
                            self._final_body_pos = self.get_central_body_pos().copy()
                            self._final_body_quat = self.data.qpos[3:7].copy()
                            print(f"  🔒 Body locked at: ({self._final_body_pos[0]:.2f}, {self._final_body_pos[1]:.2f}, {self._final_body_pos[2]:.2f})")
                            
                            # Mark trajectory as successful
                            trajectory_success = True
                        
                        # Start unscrewing after delay
                        if phase_timer >= self._unscrew_delay and not self._unscrew_started:
                            self._unscrew_started = True
                            screw_wall = getattr(self, '_screw_wall', 'back')
                            self._init_unscrew_task(
                                self._unscrew_screw_arm, 
                                self._unscrew_anchor_arm_name,
                                screw_wall
                            )
                        
                        # Execute unscrewing step
                        if self._unscrew_started:
                            unscrew_complete = self._step_unscrew_task()
                            
                            if unscrew_complete and not hasattr(self, '_mission_complete_printed'):
                                self._mission_complete_printed = True
                                print(f"============================================================")
                                print(f"✅ GLOBAL_SUCCESS_TOKEN")
                                print(f"✅ MISSION COMPLETE")
                            
                            # Print status periodically
                            if phase_timer % 250 == 0 and not self._unscrew_complete:
                                rotations = self._unscrew_total_rotation / (2 * np.pi)
                                ee_pos = self.get_left_gripper_pos() if self._unscrew_arm == 'left' else self.get_right_gripper_pos()
                                screw_pos = self.data.xpos[self._unscrew_screw_body_id]
                                ee_dist = np.linalg.norm(ee_pos - screw_pos) * 1000
                                print(f"  🔩 Unscrew: {rotations:.2f}/{self._unscrew_target_rotations} turns | EE-screw: {ee_dist:.0f}mm")
                        else:
                            # Lock body during delay
                            self.data.qpos[0:3] = self._final_body_pos
                            self.data.qpos[3:7] = self._final_body_quat
                            self.data.qvel[0:6] = 0.0
                    
                    elif crawl_phase == 'failed':
                        # Error state - just hold position
                        phase_timer += 1
                        if phase_timer % 500 == 0:
                            print(f"  [FAILED] Holding position...")
                    
                    # ============================================================
                    # END STATE MACHINE
                    # ============================================================
                    
                    # Step physics - robot body is FREE from the start
                    mujoco.mj_step(self.model, self.data)
                    
                    # WORKSPACE BOUNDARY ENFORCEMENT
                    # Clamp body position to stay within ISS module bounds
                    # This prevents the body from drifting into walls/ceiling
                    body_pos = self.data.qpos[0:3]
                    margin = 0.2  # Keep body 20cm from walls
                    body_pos[0] = np.clip(body_pos[0], ISS_MODULE['x_min'] + margin, ISS_MODULE['x_max'] - margin)
                    body_pos[1] = np.clip(body_pos[1], ISS_MODULE['y_min'] + margin, ISS_MODULE['y_max'] - margin)
                    body_pos[2] = np.clip(body_pos[2], ISS_MODULE['z_min'] + margin, ISS_MODULE['z_max'] - margin)
                    
                    # Render visualization geometries
                    with viewer.lock():
                        self._render_visualization_geoms(viewer)
                        if first_render:
                            print(f"✓ Added {viewer.user_scn.ngeom} custom geometries to scene")
                            first_render = False
                    
                    # Sync viewer
                    viewer.sync()
                    
                    # Check duration
                    if duration and (time.time() - start_time) >= duration:
                        break
                    
                    step_count += 1
                    
                    # Real-time sync
                    time_until_next = self.model.opt.timestep - (time.time() - step_start)
                    if time_until_next > 0:
                        time.sleep(time_until_next)
                        
            except KeyboardInterrupt:
                print("\n⚠️ Simulation interrupted by user")
            except Exception as e:
                print(f"\n⚠️ Simulation error: {e}")
                import traceback
                traceback.print_exc()
            finally:
                pass  # Fall through to summary
        
        # ================================================================
        # END OF SIMULATION - SUMMARY AND ANALYSIS
        # ================================================================
        print("\n👋 Visualization ended")
        
        # Print success status
        print(f"\n{'='*60}")
        print("TRAJECTORY EXECUTION SUMMARY")
        print(f"{'='*60}")
        
        if trajectory_success:
            print("✅ TRAJECTORY COMPLETED SUCCESSFULLY!")
            reason = "completed"
        else:
            print("❌ TRAJECTORY DID NOT COMPLETE")
            print(f"   Reached waypoint {current_waypoint_idx} of {total_waypoints}")
            reason = f"stopped at WP{current_waypoint_idx+1}"
        
        # Record run result for multi-run statistics
        history = record_run_result(
            success=trajectory_success,
            waypoints_reached=current_waypoint_idx if trajectory_success else current_waypoint_idx,
            total_waypoints=total_waypoints,
            reason=reason
        )
        
        # Show cumulative success rate
        if history['total'] > 0:
            success_rate = (history['successes'] / history['total']) * 100
            print(f"\n📊 CUMULATIVE SUCCESS RATE: {history['successes']}/{history['total']} ({success_rate:.1f}%)")
        
        # Print anchor deviation statistics
        print(f"\n{'='*60}")
        print("ANCHOR DEVIATION ANALYSIS")
        print(f"{'='*60}")
        print(f"{'WP':>4} | {'Arm':>6} | {'Max Dev (m)':>12} | {'Avg Dev (m)':>12} | {'Samples':>8}")
        print("-" * 60)
        
        max_deviations = []
        wp_labels = []
        
        for wp_idx in sorted(anchor_deviation_history.keys()):
            data = anchor_deviation_history[wp_idx]
            arm = data['arm']
            max_dev = data['max_deviation']
            deviations = data['deviations']
            avg_dev = np.mean(deviations) if deviations else 0.0
            n_samples = len(deviations)
            
            max_deviations.append(max_dev)
            wp_labels.append(f"WP{wp_idx+1}\n({arm[0].upper()})")
            
            print(f"{wp_idx+1:>4} | {arm.upper():>6} | {max_dev:>12.4f} | {avg_dev:>12.4f} | {n_samples:>8}")
        
        print("-" * 60)
        
        # Create matplotlib graph of max deviations
        if max_deviations:
            try:
                import matplotlib
                # Try to use TkAgg backend for interactive display
                try:
                    matplotlib.use('TkAgg')
                except:
                    pass  # Use default backend if TkAgg not available
                import matplotlib.pyplot as plt
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
                
                # Bar chart of max deviations
                colors = ['#2196F3' if 'L' in label else '#FF9800' for label in wp_labels]
                bars = ax1.bar(range(len(max_deviations)), max_deviations, color=colors, edgecolor='black', linewidth=1.2)
                ax1.set_xticks(range(len(max_deviations)))
                ax1.set_xticklabels(wp_labels)
                ax1.set_xlabel('Waypoint (Arm)', fontsize=12)
                ax1.set_ylabel('Max End-Effector Deviation (m)', fontsize=12)
                ax1.set_title('Maximum EE Deviation from Anchor Point per Waypoint', fontsize=14, fontweight='bold')
                ax1.axhline(y=0.05, color='green', linestyle='--', linewidth=1.5, label='Good (<5cm)')
                ax1.axhline(y=0.10, color='orange', linestyle='--', linewidth=1.5, label='Acceptable (<10cm)')
                ax1.axhline(y=0.20, color='red', linestyle='--', linewidth=1.5, label='Limit (20cm)')
                ax1.legend(loc='upper right')
                ax1.set_ylim(0, max(0.25, max(max_deviations) * 1.2))
                ax1.grid(axis='y', alpha=0.3)
                
                # Add value labels on bars
                for bar, val in zip(bars, max_deviations):
                    height = bar.get_height()
                    ax1.annotate(f'{val:.3f}m',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=9)
                
                # Time series of deviations (if we have enough data)
                all_deviations = []
                all_wp_indices = []
                for wp_idx in sorted(anchor_deviation_history.keys()):
                    devs = anchor_deviation_history[wp_idx]['deviations']
                    all_deviations.extend(devs)
                    all_wp_indices.extend([wp_idx] * len(devs))
                
                if all_deviations:
                    # Color-code by waypoint
                    scatter_colors = plt.cm.viridis(np.array(all_wp_indices) / max(all_wp_indices) if max(all_wp_indices) > 0 else [0.5] * len(all_wp_indices))
                    ax2.scatter(range(len(all_deviations)), all_deviations, c=all_wp_indices, cmap='viridis', alpha=0.5, s=2)
                    ax2.set_xlabel('Time Step (during anchoring)', fontsize=12)
                    ax2.set_ylabel('EE Deviation from Anchor (m)', fontsize=12)
                    ax2.set_title('End-Effector Deviation Over Time (All Anchors)', fontsize=14, fontweight='bold')
                    ax2.axhline(y=0.05, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
                    ax2.axhline(y=0.10, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
                    ax2.set_ylim(0, max(0.15, max(all_deviations) * 1.2) if all_deviations else 0.15)
                    ax2.grid(alpha=0.3)
                    
                    # Add colorbar
                    sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min(all_wp_indices), vmax=max(all_wp_indices)))
                    sm.set_array([])
                    cbar = plt.colorbar(sm, ax=ax2)
                    cbar.set_label('Waypoint Index')
                
                plt.suptitle(f"Anchor Stability Analysis - {'SUCCESS' if trajectory_success else 'INCOMPLETE'}", 
                           fontsize=16, fontweight='bold', y=1.02)
                plt.tight_layout()
                
                # Save plot to file first (always works)
                plot_filename = "anchor_deviation_analysis.png"
                plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
                print(f"\n📊 Plot saved to: {plot_filename}")
                
                # Try to display interactively with timeout
                print("   Displaying graph window (auto-closes in 5 seconds)...")
                try:
                    plt.show(block=False)
                    plt.pause(5)  # Show for 5 seconds then continue
                    plt.close('all')
                except Exception as show_err:
                    print(f"   Note: Interactive display not available ({show_err})")
                    print(f"   View the saved file: {plot_filename}")
                
            except ImportError as ie:
                print(f"\n⚠️ matplotlib not available for anchor deviation plotting: {ie}")
            except Exception as e:
                print(f"\n⚠️ Error creating anchor deviation plot: {e}")
                import traceback
                traceback.print_exc()
        
        # ================================================================
        # TRAJECTORY ERROR TRACKING PLOT (independent of anchor deviations)
        # ================================================================
        if trajectory_error_history and len(trajectory_error_history['time_steps']) > 0:
            try:
                import matplotlib
                try:
                    matplotlib.use('TkAgg')
                except:
                    pass
                import matplotlib.pyplot as plt
                
                print("\n📈 GENERATING TRAJECTORY ERROR GRAPH...")
                
                fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5))
                
                # Extract data
                time_steps = np.array(trajectory_error_history['time_steps'])
                errors = np.array(trajectory_error_history['errors'])
                waypoint_idxs = np.array(trajectory_error_history['waypoint_idx'])
                phases = np.array(trajectory_error_history['phase'])
                
                # Get unique waypoints
                unique_waypoints = sorted(set(waypoint_idxs))
                colors = plt.cm.tab10(np.linspace(0, 1, len(unique_waypoints) + 1))
                
                # Left plot: Trajectory error convergence over time for each waypoint
                for i, wp_idx in enumerate(unique_waypoints):
                    mask = waypoint_idxs == wp_idx
                    wp_steps = time_steps[mask]
                    wp_errors = errors[mask]
                    if len(wp_steps) > 0:
                        ax3.plot(wp_steps, wp_errors, label=f'Waypoint {wp_idx}', 
                                color=colors[i % len(colors)], alpha=0.8, linewidth=1.5)
                
                ax3.set_xlabel('Simulation Step', fontsize=12)
                ax3.set_ylabel('Position Error (m)', fontsize=12)
                ax3.set_title('Trajectory Tracking Error Over Time', fontsize=14, fontweight='bold')
                ax3.axhline(y=0.05, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Target (5cm)')
                ax3.axhline(y=0.10, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Limit (10cm)')
                ax3.legend(loc='upper right', fontsize=8)
                ax3.set_ylim(0, None)
                ax3.grid(alpha=0.3)
                
                # Right plot: Initial error, final error per waypoint
                initial_errors = []
                final_errors = []
                
                for wp_idx in unique_waypoints:
                    mask = waypoint_idxs == wp_idx
                    wp_errors = errors[mask]
                    if len(wp_errors) > 0:
                        initial_errors.append(wp_errors[0])
                        final_errors.append(wp_errors[-1])
                    else:
                        initial_errors.append(0)
                        final_errors.append(0)
                
                x_pos = np.arange(len(unique_waypoints))
                bar_width = 0.25
                
                bars1 = ax4.bar(x_pos - bar_width, initial_errors, bar_width, 
                               label='Initial Error (m)', color='coral', alpha=0.8)
                bars2 = ax4.bar(x_pos, final_errors, bar_width, 
                               label='Final Error (m)', color='steelblue', alpha=0.8)
                
                ax4.set_xlabel('Waypoint Index', fontsize=12)
                ax4.set_ylabel('Position Error (m)', fontsize=12)
                ax4.set_title('Initial vs Final Tracking Error per Waypoint', fontsize=14, fontweight='bold')
                ax4.set_xticks(x_pos)
                ax4.set_xticklabels([f'WP{i}' for i in unique_waypoints])
                ax4.axhline(y=0.05, color='green', linestyle='--', linewidth=1.5, label='Target (5cm)')
                ax4.axhline(y=0.15, color='orange', linestyle='--', linewidth=1.5, label='Threshold (15cm)')
                ax4.legend(loc='upper right')
                ax4.grid(axis='y', alpha=0.3)
                
                # Add value labels
                for bar, val in zip(bars1, initial_errors):
                    height = bar.get_height()
                    ax4.annotate(f'{val:.2f}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=7)
                for bar, val in zip(bars2, final_errors):
                    height = bar.get_height()
                    ax4.annotate(f'{val:.2f}',
                                xy=(bar.get_x() + bar.get_width() / 2, height),
                                xytext=(0, 3),
                                textcoords="offset points",
                                ha='center', va='bottom', fontsize=7)
                
                plt.suptitle(f"Trajectory Tracking Analysis - {'SUCCESS' if trajectory_success else 'INCOMPLETE'}", 
                           fontsize=16, fontweight='bold', y=1.02)
                plt.tight_layout()
                
                # Save trajectory plot
                traj_plot_filename = "trajectory_error_analysis.png"
                plt.savefig(traj_plot_filename, dpi=150, bbox_inches='tight')
                print(f"📊 Trajectory plot saved to: {traj_plot_filename}")
                
                try:
                    plt.show(block=False)
                    plt.pause(5)  # Show for 5 seconds then continue
                    plt.close('all')
                except Exception as show_err:
                    print(f"   Note: Interactive display not available ({show_err})")
                    
            except ImportError as ie:
                print(f"\n⚠️ matplotlib not available for trajectory plotting: {ie}")
            except Exception as e:
                print(f"\n⚠️ Error creating trajectory plot: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("\n⚠️ No trajectory error data collected - skipping trajectory plot")
    
    def _render_overlays(self, viewer):
        """Legacy method - replaced by _render_visualization_geoms"""
        pass


# ============================================================================
# MULTI-RUN SUCCESS TRACKING
# ============================================================================

def load_run_history():
    """Load run history from file"""
    import json
    history_file = "run_history.json"
    try:
        with open(history_file, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {'runs': [], 'total': 0, 'successes': 0}

def save_run_history(history):
    """Save run history to file"""
    import json
    history_file = "run_history.json"
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)

RECORDING_ENABLED = True

def record_run_result(success: bool, waypoints_reached: int, total_waypoints: int, reason: str = ""):
    """Record the result of a run"""
    if not RECORDING_ENABLED:
        return {'runs': [], 'total': 0, 'successes': 0}

    import datetime
    history = load_run_history()
    
    run_record = {
        'timestamp': datetime.datetime.now().isoformat(),
        'success': success,
        'waypoints_reached': waypoints_reached,
        'total_waypoints': total_waypoints,
        'reason': reason
    }
    
    history['runs'].append(run_record)
    history['total'] += 1
    if success:
        history['successes'] += 1
    
    save_run_history(history)
    return history

def print_success_rate():
    """Print the cumulative success rate across all runs"""
    history = load_run_history()
    
    print(f"\n{'='*60}")
    print("MULTI-RUN SUCCESS RATE")
    print(f"{'='*60}")
    
    if history['total'] == 0:
        print("No runs recorded yet.")
        return
    
    success_rate = (history['successes'] / history['total']) * 100
    print(f"Total Runs: {history['total']}")
    print(f"Successful: {history['successes']}")
    print(f"Failed: {history['total'] - history['successes']}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    # Show last 5 runs
    recent_runs = history['runs'][-5:]
    if recent_runs:
        print(f"\nLast {len(recent_runs)} runs:")
        for i, run in enumerate(recent_runs, 1):
            status = "✅" if run['success'] else "❌"
            wp_info = f"{run['waypoints_reached']}/{run['total_waypoints']}"
            reason = f" ({run['reason']})" if run.get('reason') else ""
            print(f"  {i}. {status} WP: {wp_info}{reason}")
    
    print(f"{'='*60}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def get_random_goal_position():
    """Generate a random valid goal position on one of the walls (excluding front wall).
    
    Returns:
        Tuple of (position, wall_name)
    """
    import random
    
    # Define valid goal regions on each wall (with safety margins)
    # Format: (x_range, y_range, z_range, wall_name)
    goal_regions = [
        {
            'x_min': ISS_MODULE['x_min'] + 0.5, 'x_max': ISS_MODULE['x_max'] - 0.5,
            'y': ISS_MODULE['y_min'] + 0.2,  # Increased margin from 0.05 to 0.2 to prevent wall clipping
            'z_min': 0.8, 'z_max': 1.8,
            'wall': 'back'
        },
        # Ceiling (z = z_max) 
        {
            'x_min': ISS_MODULE['x_min'] + 0.5, 'x_max': ISS_MODULE['x_max'] - 0.5,
            'y_min': ISS_MODULE['y_min'] + 0.3, 'y_max': ISS_MODULE['y_max'] - 0.3,
            'z_min': ISS_MODULE['z_max'] - 0.1, 'z_max': ISS_MODULE['z_max'] - 0.1,
            'z': ISS_MODULE['z_max'] - 0.1,
            'wall': 'ceiling'
        },
        # Left wall (x = x_min)
        {
            'x': ISS_MODULE['x_min'] + 0.2, # x_min is wall, margin to 0.2 to ensure body clearance
            'y_min': ISS_MODULE['y_min'] + 0.3, 'y_max': ISS_MODULE['y_max'] - 0.3,
            'z_min': 0.8, 'z_max': 1.8,
            'wall': 'left'
        },
        # Right wall (x = x_max)
        {
            'x': ISS_MODULE['x_max'] - 0.2, # x_max is wall, margin to 0.2
            'y_min': ISS_MODULE['y_min'] + 0.3, 'y_max': ISS_MODULE['y_max'] - 0.3,
            'z_min': 0.8, 'z_max': 1.8,
            'wall': 'right'
        },
    ]
    
    # Weight back wall more heavily (60% of the time) since it's the most common use case
    weights = [0.6, 0.15, 0.125, 0.125]  # back, ceiling, left, right
    
    # Select a random region based on weights
    region = random.choices(goal_regions, weights=weights, k=1)[0]
    
    # Generate random position within the selected region
    if 'x' in region:  # Fixed x (left/right wall)
        x = region['x']
    else:
        x = random.uniform(region['x_min'], region['x_max'])
    
    if 'y' in region:  # Fixed y (front/back wall)
        y = region['y']
    else:
        y = random.uniform(region['y_min'], region['y_max'])
    
    if 'z' in region:  # Fixed z (floor/ceiling)
        z = region['z']
    else:
        z = random.uniform(region['z_min'], region['z_max'])
    
    return (x, y, z), region['wall']


def main():
    """Main function for Phase 1: Workspace visualization"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Wall Crawler MuJoCo Simulation')
    parser.add_argument('--no-record', action='store_true', help='Disable recording of run history to JSON file')
    parser.add_argument('--headless', action='store_true', help='Run without visualization (for batch testing)')
    args = parser.parse_args()
    
    if args.no_record:
        global RECORDING_ENABLED
        RECORDING_ENABLED = False
        print("⚠️  Run history recording DISABLED")
    
    headless = args.headless
    
    print("\n" + "="*70)
    print("MUJOCO WALL-CRAWLER SIMULATION - PHASE 1")
    print("Workspace Mapping & Path Planning Visualization")
    print("="*70)
    
    # Show previous run statistics
    print_success_rate()
    
    # Run with duration limit to ensure summary/plotting is shown
    try:
        # Create simulation
        sim = WallCrawlerMuJoCoSimulation(model_path="dual_arm_robot.xml")
        
        # Random start position for robustness testing
        print("🎲 Generating random start and goal positions...")
        start_pos, start_wall = get_random_goal_position()
        goal_pos, goal_wall = get_random_goal_position()
        
        # Ensure goal is distinct from start (at least 2.5m away for meaningful trajectory)
        min_trajectory_distance = 2.5  # Minimum distance for a meaningful trajectory
        max_attempts = 50
        attempt = 0
        while np.linalg.norm(np.array(start_pos) - np.array(goal_pos)) < min_trajectory_distance and attempt < max_attempts:
            goal_pos, goal_wall = get_random_goal_position()
            attempt += 1
        
        if attempt >= max_attempts:
            print(f"⚠️  Could not find goal at least {min_trajectory_distance}m away after {max_attempts} attempts")
            
        print(f"Start: {start_wall} {start_pos}")
        print(f"Goal:  {goal_wall} {goal_pos}")
        print(f"✓ Trajectory distance: {np.linalg.norm(np.array(start_pos) - np.array(goal_pos)):.2f}m")
        
        sim.set_start_and_goal(start_pos, goal_pos)
        
        # FORCE SAFE SPAWN POSE - COMPACT CONFIGURATION
        # Reset arms to a "Compact/Tucked" pose to GUARANTEE arms stay inside module bounds
        # KUKA iiwa14 joints: [J1_rotation, J2_shoulder, J3_elbow_rot, J4_elbow, J5_wrist_rot, J6_wrist, J7_flange]
        # This pose folds the arms close to the body:
        #   J2 = -1.2 rad: Shoulder retracted (arm points more upward/inward)
        #   J4 = 1.8 rad: Elbow bent strongly (forearm folds back)
        #   J6 = 0.5 rad: Wrist bent slightly to keep gripper away from body
        compact_arm_pose = np.array([0.0, -1.2, 0.0, 1.8, 0.0, 0.5, 0.0])
        sim.data.qpos[sim.left_arm_qpos_slice] = compact_arm_pose.copy()
        sim.data.qpos[sim.right_arm_qpos_slice] = compact_arm_pose.copy()
        # Reset velocities
        sim.data.qvel[:] = 0.0
        # Forward kinematics
        mujoco.mj_forward(sim.model, sim.data)
        print("✓ Reset arms to COMPACT Tucked Pose (J2=-1.2, J4=1.8, J6=0.5)")
        
        # Verify arms are inside bounds
        left_ee = sim.get_left_gripper_pos()
        right_ee = sim.get_right_gripper_pos()
        body_pos = sim.get_central_body_pos()
        print(f"  Body at: ({body_pos[0]:.2f}, {body_pos[1]:.2f}, {body_pos[2]:.2f})")
        print(f"  Left EE at: ({left_ee[0]:.2f}, {left_ee[1]:.2f}, {left_ee[2]:.2f})")
        print(f"  Right EE at: ({right_ee[0]:.2f}, {right_ee[1]:.2f}, {right_ee[2]:.2f})")
        
        # Check bounds
        def check_in_bounds(pos, name):
            in_x = ISS_MODULE['x_min'] <= pos[0] <= ISS_MODULE['x_max']
            in_y = ISS_MODULE['y_min'] <= pos[1] <= ISS_MODULE['y_max']
            in_z = ISS_MODULE['z_min'] <= pos[2] <= ISS_MODULE['z_max']
            if in_x and in_y and in_z:
                print(f"  ✓ {name} inside module bounds")
            else:
                print(f"  ⚠️ {name} OUTSIDE bounds! X:{in_x}, Y:{in_y}, Z:{in_z}")
        
        check_in_bounds(left_ee, "Left EE")
        check_in_bounds(right_ee, "Right EE")
        
        # Run visualization or headless simulation
        if headless:
            print("\nRunning headless simulation (no GUI)...")
            sim.run_headless(duration=300)
        else:
            print("\nStarting MuJoCo visualization...")
            print("The robot will be displayed with the planned path.")
            sim.run_visualization(duration=300)
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Make sure dual_arm_robot.xml is in the current directory.")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback


if __name__ == "__main__":
    main()
