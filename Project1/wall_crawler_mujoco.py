#!/usr/bin/env python3
"""
MuJoCo-Based Dual-Arm Wall-Crawler Simulation

This simulation uses the dual_arm_robot.xml MuJoCo environment to implement
the wall-crawling algorithm from wall_crawler_iss_module.py.

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

# DUAL ARM CONFIGURATION
LEFT_ARM_OFFSET = -0.25   # X offset from central body
RIGHT_ARM_OFFSET = 0.25   # X offset from central body
ARM_SEPARATION = 0.5      # 0.5m between arms

# KUKA iiwa14 workspace
# KUKA spec: 0.82m reach from A1 axis (arm only)
# With Robotiq 2F85 gripper attached: ~1.25m total reach from A1 to gripper center
# For path planning we use full reach
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
    
    def __init__(self, step_size: float = 0.5, arm_reach: float = KUKA_REACH):
        self.step_size = step_size
        self.arm_reach = arm_reach
        
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
        """Get reachable neighbor states from current state"""
        neighbors = []
        current_pos = np.array(state.position)
        current_wall = state.wall
        
        # The arm that will move next (alternating)
        next_arm = 'right' if state.active_arm == 'left' else 'left'
        
        for (pos, wall), _ in zip(self.wall_positions, range(len(self.wall_positions))):
            pos_arr = np.array(pos)
            distance = np.linalg.norm(pos_arr - current_pos)
            
            # Check if within arm reach
            if distance <= self.arm_reach and distance > 0.1:
                # Same wall or adjacent wall transitions
                if wall == current_wall or self._can_transition(current_wall, wall):
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
        """Heuristic: Euclidean distance to goal"""
        return self.mapper.get_distance(state.position, goal_pos)
    
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
                return self._reconstruct_path(current)
            
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
        
        # Initialize wall position mapper with SMALLER step size for better reachability
        self.wall_mapper = WallPositionMapper(step_size=0.3, arm_reach=KUKA_REACH)
        
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
        
        # Controller gains - Balanced for stable tracking
        self.kp_position = 150.0      # Position control
        self.kd_position = 40.0       # Damping
        self.kp_orientation = 40.0    # Orientation control
        self.kd_orientation = 20.0
        self.lambda_dls = 0.08        # Damping for stability
        
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
        
        # Compute path using snapped positions
        self.current_path = self.path_planner.find_path(snapped_start, snapped_goal, start_arm='left')
        
        if self.current_path:
            print(f"\n✓ Path found with {len(self.current_path)} steps:")
            for i, state in enumerate(self.current_path):
                print(f"  Step {i+1}: {state.wall:10s} | "
                      f"({state.position[0]:6.2f}, {state.position[1]:6.2f}, {state.position[2]:6.2f}) | "
                      f"{state.active_arm.upper()} arm")
            
            self._transition_state(RobotState.READY_TO_MOVE)
        else:
            print("✗ No path found!")
            self._transition_state(RobotState.ERROR)
    
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
    
    def compute_jacobian(self, body_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the Jacobian for a given body"""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, body_id)
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
    
    def compute_arm_control(self, arm: str = 'left') -> Tuple[np.ndarray, np.ndarray]:
        """Compute Cartesian space control for specified arm
        
        Returns:
            (joint_position_command, position_error)
        """
        if arm == 'left':
            current_pos = self.get_left_gripper_pos()
            current_orient = self.get_left_ee_orientation()
            target_pos = self.left_target_position
            body_id = self.left_ee_body_id
            qpos_slice = self.left_arm_qpos_slice
            qvel_slice = self.left_arm_qvel_slice
        else:
            current_pos = self.get_right_gripper_pos()
            current_orient = self.get_right_ee_orientation()
            target_pos = self.right_target_position
            body_id = self.right_ee_body_id
            qpos_slice = self.right_arm_qpos_slice
            qvel_slice = self.right_arm_qvel_slice
        
        if target_pos is None:
            return self.data.qpos[qpos_slice].copy(), np.zeros(3)
        
        # Position error only (ignore orientation for now)
        pos_error = target_pos - current_pos
        
        # Get position Jacobian only
        jacp, jacr = self.compute_jacobian(body_id)
        
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
        
        # Add velocity damping
        joint_vel_cmd -= self.kd_position * current_joint_vel
        
        # Convert to position command
        current_joint_pos = self.data.qpos[qpos_slice]
        joint_pos_cmd = current_joint_pos + joint_vel_cmd * self.model.opt.timestep
        
        return joint_pos_cmd, pos_error
    
    def apply_arm_control(self, arm: str = 'left'):
        """Apply arm control to reach target position"""
        joint_cmd, pos_error = self.compute_arm_control(arm)
        
        if arm == 'left':
            self.data.ctrl[self.left_arm_actuator_slice] = joint_cmd
        else:
            self.data.ctrl[self.right_arm_actuator_slice] = joint_cmd
        
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
            print(f"  🔓 LEFT arm RELEASED")
        else:
            self.right_arm_anchored = False
            self.right_anchor_position = None
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
        
        # 5. Draw path waypoints and lines
        if self.current_path:
            for i, state in enumerate(self.current_path):
                # Draw waypoint sphere
                if i == 0:
                    color = GREEN  # Start
                elif i == len(self.current_path) - 1:
                    color = RED  # Goal
                elif state.active_arm == 'left':
                    color = ORANGE  # Left arm waypoint
                else:
                    color = PURPLE  # Right arm waypoint
                
                self._add_marker_geom(scene, state.position, size=0.06, rgba=color)
                
                # Draw line to next waypoint
                if i < len(self.current_path) - 1:
                    next_state = self.current_path[i + 1]
                    self._add_line_geom(scene, state.position, next_state.position,
                                       size=0.02, rgba=CYAN)
            
            # Highlight current target (if executing path)
            if 0 <= self.current_path_index < len(self.current_path):
                target = self.current_path[self.current_path_index]
                self._add_marker_geom(scene, target.position, size=0.12, rgba=YELLOW)
    
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
        print("  🟢 Green sphere: Start position")
        print("  🔴 Red sphere: Goal position")
        print("  🔵 Blue/Orange spheres: Path waypoints (left/right arm)")
        print("  🟡 Yellow sphere: Current target")
        print(f"  🔵 Blue transparent sphere: Left arm workspace ({WORKSPACE_SPHERE_RADIUS}m from elbow)")
        print(f"  🟠 Orange transparent sphere: Right arm workspace ({WORKSPACE_SPHERE_RADIUS}m from elbow)")
        print("  ⎯⎯ Cyan lines: Path connections")
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
            
            if self.start_position and self.current_path and len(self.current_path) > 1:
                # First waypoint (left arm anchor)
                wp1 = np.array(self.current_path[0].position)
                # Second waypoint (right arm target)
                wp2 = np.array(self.current_path[1].position)
                
                # Position body between the two target Z heights
                # wp1 Z = 1.10, wp2 Z = 1.40, so body at Z = 1.25 (midpoint)
                # This way left arm reaches down, right arm reaches up - symmetric
                body_x = (wp1[0] + wp2[0]) / 2  # Center X between waypoints
                body_y = wp1[1] - 0.55  # 55cm back from the front wall
                body_z = (wp1[2] + wp2[2]) / 2  # Midpoint Z between targets (1.25)
                
                # Clamp to ensure body is INSIDE the module with good margin
                body_x = np.clip(body_x, ISS_MODULE['x_min'] + 0.5, ISS_MODULE['x_max'] - 0.5)
                body_y = np.clip(body_y, ISS_MODULE['y_min'] + 0.4, ISS_MODULE['y_max'] - 0.6)
                body_z = np.clip(body_z, ISS_MODULE['z_min'] + 0.5, ISS_MODULE['z_max'] - 0.5)
                
                self.data.qpos[0] = body_x
                self.data.qpos[1] = body_y
                self.data.qpos[2] = body_z
                
                print(f"\n📍 ISS MODULE CENTER: ({module_center_x:.2f}, {module_center_y:.2f}, {module_center_z:.2f})")
                print(f"  Initial body position: ({body_x:.2f}, {body_y:.2f}, {body_z:.2f})")
                print(f"  ISS module bounds: X[{ISS_MODULE['x_min']:.1f}, {ISS_MODULE['x_max']:.1f}], Y[{ISS_MODULE['y_min']:.1f}, {ISS_MODULE['y_max']:.1f}], Z[{ISS_MODULE['z_min']:.1f}, {ISS_MODULE['z_max']:.1f}]")
                print(f"  WP1 (left anchor): {wp1}")
                print(f"  WP2 (right target): {wp2}")
            elif self.start_position:
                anchor_x, anchor_y, anchor_z = self.start_position
                self.data.qpos[0] = anchor_x
                self.data.qpos[1] = anchor_y - 0.6
                self.data.qpos[2] = anchor_z - 0.2
            else:
                self.data.qpos[0] = 1.0
                self.data.qpos[1] = 0.6
                self.data.qpos[2] = 1.0
            
            self.data.qpos[3] = 1.0   # quat w
            self.data.qpos[4:7] = 0.0 # quat xyz
            
            # LEFT ARM - Need to reach FORWARD from Y=1.10 to Y=1.65, DOWN from Z=1.25 to Z=1.10
            # The RIGHT arm configuration worked with joint2=-1.2, joint4=0.8
            # For LEFT arm, joint2 should be positive (same pitch direction)
            # and joint4 should be negative (opposite elbow bend to reach DOWN instead of UP)
            self.data.qpos[7] = 0.0      # joint1 - no rotation
            self.data.qpos[8] = 1.2      # joint2 - pitch forward (positive for left)
            self.data.qpos[9] = 0.0      # joint3 
            self.data.qpos[10] = -0.8    # joint4 - elbow bent DOWN (negative = down)
            self.data.qpos[11] = 0.0     # joint5 
            self.data.qpos[12] = -0.5    # joint6 - wrist angled down (opposite of right arm)
            self.data.qpos[13] = 0.0     # joint7 
            
            # RIGHT ARM - This configuration WORKS! (reached target with 0.050m error)
            # Keep it exactly the same
            self.data.qpos[22] = 0.0     # joint1 - no rotation
            self.data.qpos[23] = -1.2    # joint2 - pitch forward (negative for right arm)
            self.data.qpos[24] = 0.0     # joint3 
            self.data.qpos[25] = 0.8     # joint4 - elbow bent UP (positive for right arm)
            self.data.qpos[26] = 0.0     # joint5 
            self.data.qpos[27] = -0.5    # joint6 - wrist angled up
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
            
            # Phase 2: Control variables - START ALREADY ANCHORED
            settling_steps = 10        # Very short settling
            settling_timeout = 1000    # Give more time for left arm to reach anchor
            arm_control_enabled = True  # Enable arm control immediately
            
            # Start directly in moving_right phase since left arm is already anchored
            crawl_phase = 'settling'
            phase_timer = 0
            
            # Set right arm target immediately if we have a path
            if self.current_path and len(self.current_path) > 1:
                second_wp = self.current_path[1]
                self.right_target_position = np.array(second_wp.position)
                print(f"✓ Right arm target set to: {second_wp.position}")
            
            # ================================================================
            # PRE-POSITIONING CHECK: Verify body is properly inside the module
            # DO NOT move the body - keep it at the center position
            # Let IK move the arms to reach the targets
            # ================================================================
            print("\n⚙️  Verifying robot position...")
            
            # Run forward kinematics to get current positions
            mujoco.mj_forward(self.model, self.data)
            
            # Report positions after positioning
            body_pos = self.get_central_body_pos()
            left_pos = self.get_left_gripper_pos()
            right_pos = self.get_right_gripper_pos()
            anchor = self.left_anchor_position
            
            print(f"  Body at ({body_pos[0]:.2f}, {body_pos[1]:.2f}, {body_pos[2]:.2f})")
            print(f"  Left gripper at ({left_pos[0]:.2f}, {left_pos[1]:.2f}, {left_pos[2]:.2f})")
            print(f"  Right gripper at ({right_pos[0]:.2f}, {right_pos[1]:.2f}, {right_pos[2]:.2f})")
            if anchor is not None:
                left_to_anchor = np.linalg.norm(anchor - left_pos)
                print(f"  Left gripper to anchor distance: {left_to_anchor:.3f}m")
                if left_to_anchor > KUKA_REACH:
                    print(f"  ⚠️ WARNING: Anchor may be out of reach (> {KUKA_REACH}m)")
            
            # Verify body is inside the module
            if not (ISS_MODULE['x_min'] < body_pos[0] < ISS_MODULE['x_max'] and
                    ISS_MODULE['y_min'] < body_pos[1] < ISS_MODULE['y_max'] and
                    ISS_MODULE['z_min'] < body_pos[2] < ISS_MODULE['z_max']):
                print(f"  ⚠️ WARNING: Body position outside ISS module!")
            else:
                print(f"  ✓ Body position is INSIDE ISS module")
            
            # DO NOT lock body yet - let left arm use IK to reach anchor first
            # Locking happens in state machine after settling phase
            
            print(f"\n📍 Visualizing {len(self.wall_mapper.wall_positions)} wall grip positions...")
            print(f"📍 Workspace spheres: Left (blue) and Right (orange) with {WORKSPACE_SPHERE_RADIUS}m radius (centered at elbow)")
            if self.current_path:
                print(f"📍 Showing path with {len(self.current_path)} waypoints")
            
            print(f"\n{'='*60}")
            print("PHASE 2: CONTINUOUS ARM MOVEMENT")
            print(f"{'='*60}")
            print("Left arm already anchored, right arm moving to second waypoint...")
            
            while viewer.is_running():
                step_start = time.time()
                
                # ============================================================
                # WALL CRAWLING STATE MACHINE - SIMPLIFIED
                # Phase 1: Left arm uses IK to reach anchor
                # Phase 2: Lock body, Right arm uses IK to reach target
                # ============================================================
                
                if self.current_path and len(self.current_path) > 0:
                    
                    # PHASE: SETTLING - Use IK to position left arm at anchor
                    if crawl_phase == 'settling':
                        phase_timer += 1
                        # Keep body fixed during settling (only arms move)
                        # Store initial body position on first step
                        if phase_timer == 1:
                            self.settling_body_pos = self.data.qpos[0:3].copy()
                            self.settling_body_quat = self.data.qpos[3:7].copy()
                        
                        # Restore body position each step to keep it fixed
                        self.data.qpos[0:3] = self.settling_body_pos.copy()
                        self.data.qpos[3:7] = self.settling_body_quat.copy()
                        self.data.qvel[0:6] = 0.0  # Zero body velocity
                        
                        # Use IK to move left arm toward anchor
                        if self.left_anchor_position is not None:
                            self.left_target_position = self.left_anchor_position
                            left_error = self.apply_arm_control('left')
                            
                            # Print progress periodically
                            if phase_timer % 50 == 0:
                                left_pos = self.get_left_gripper_pos()
                                print(f"    [Settling {phase_timer}] Left arm error: {left_error:.3f}m at ({left_pos[0]:.2f}, {left_pos[1]:.2f}, {left_pos[2]:.2f})")
                            
                            # Transition when left arm is close to anchor
                            if left_error < 0.08 and phase_timer >= settling_steps:
                                # LOCK body and left arm now that left arm is at anchor
                                self.lock_body_position()
                                
                                crawl_phase = 'moving_right'
                                phase_timer = 0
                                self._transition_state(RobotState.MOVING_RIGHT_ARM)
                                
                                body_pos = self.get_central_body_pos()
                                left_pos = self.get_left_gripper_pos()
                                right_pos = self.get_right_gripper_pos()
                                anchor = self.left_anchor_position
                                
                                print(f"\n✓ Left arm at anchor! Starting right arm movement.")
                                print(f"  Body at: ({body_pos[0]:.2f}, {body_pos[1]:.2f}, {body_pos[2]:.2f})")
                                print(f"  Left gripper at: ({left_pos[0]:.2f}, {left_pos[1]:.2f}, {left_pos[2]:.2f})")
                                print(f"  Left anchor: ({anchor[0]:.2f}, {anchor[1]:.2f}, {anchor[2]:.2f})")
                                print(f"  Right gripper at: ({right_pos[0]:.2f}, {right_pos[1]:.2f}, {right_pos[2]:.2f})")
                                if self.right_target_position is not None:
                                    print(f"  Right target: ({self.right_target_position[0]:.2f}, {self.right_target_position[1]:.2f}, {self.right_target_position[2]:.2f})")
                        
                        # Transition after timeout regardless
                        if phase_timer >= settling_timeout:
                            # LOCK body and left arm even if not perfectly positioned
                            self.lock_body_position()
                            
                            crawl_phase = 'moving_right'
                            phase_timer = 0
                            self._transition_state(RobotState.MOVING_RIGHT_ARM)
                            print(f"\n⚠️ Left arm settling timeout - proceeding to right arm movement")
                    
                    # PHASE: MOVING_RIGHT - Move right arm toward second waypoint
                    elif crawl_phase == 'moving_right':
                        phase_timer += 1
                        
                        # Lock left arm joints - just zero torque so they stay in place
                        # The body is locked so left arm position is fixed
                        for i in range(7):
                            self.data.ctrl[i] = 0.0
                        
                        # Apply arm control to right arm only
                        if self.right_target_position is not None:
                            error = self.apply_arm_control('right')
                            
                            # Print progress periodically
                            if phase_timer % 200 == 0:
                                right_pos = self.get_right_gripper_pos()
                                left_pos = self.get_left_gripper_pos()
                                target = self.right_target_position
                                distance = np.linalg.norm(target - right_pos)
                                print(f"    [Step {phase_timer}] R: ({right_pos[0]:.2f}, {right_pos[1]:.2f}, {right_pos[2]:.2f}) → {distance:.3f}m | L: ({left_pos[0]:.2f}, {left_pos[1]:.2f}, {left_pos[2]:.2f})")
                            
                            # Check if close enough to target
                            right_pos = self.get_right_gripper_pos()
                            distance_to_target = np.linalg.norm(self.right_target_position - right_pos)
                            if distance_to_target < 0.05:  # Within 5cm
                                print(f"\n✅ Right arm reached target! Distance: {distance_to_target:.3f}m")
                                crawl_phase = 'done'
                                self._transition_state(RobotState.GOAL_REACHED)
                        
                        # Timeout after many steps
                        if phase_timer >= 4000:
                            right_pos = self.get_right_gripper_pos()
                            final_error = np.linalg.norm(self.right_target_position - right_pos)
                            print(f"\n  Right arm timed out. Final error: {final_error:.3f}m")
                            print(f"  Right gripper at: ({right_pos[0]:.2f}, {right_pos[1]:.2f}, {right_pos[2]:.2f})")
                            crawl_phase = 'done'
                            self._transition_state(RobotState.GOAL_REACHED)
                    
                    # PHASE: DONE - Maintain both arms
                    elif crawl_phase == 'done':
                        # Just zero torques to maintain positions
                        for i in range(14):  # All arm joints
                            self.data.ctrl[i] = 0.0
                
                # Maintain anchors BEFORE physics (sets velocities to 0)
                self.step_phase2_control()
                
                # ============================================================
                # END WALL CRAWLING STATE MACHINE
                # ============================================================
                
                # Step physics - robot is now FREE to move!
                mujoco.mj_step(self.model, self.data)
                
                # CRITICAL: Enforce anchor AFTER physics step
                # Physics may have moved the body, so we must correct it
                self.step_phase2_control()
                
                # Render visualization geometries using viewer's user scene
                with viewer.lock():
                    self._render_visualization_geoms(viewer)
                    if first_render:
                        print(f"✓ Added {viewer.user_scn.ngeom} custom geometries to scene")
                        print("✓ Physics enabled - robot is free-floating!")
                        print("⏳ Robot settling for {settling_steps} steps before control...")
                        first_render = False
                
                # Sync viewer
                viewer.sync()
                
                # Print status periodically
                if step_count % 500 == 0 and step_count > 0:
                    left_pos = self.get_left_gripper_pos()
                    right_pos = self.get_right_gripper_pos()
                    body_pos = self.get_central_body_pos()
                    
                    anchor_status = "⚓ANCHORED" if self.left_arm_anchored else "floating"
                    
                    print(f"[{self.robot_state.name:15s}] ({anchor_status}) "
                          f"Body: ({body_pos[0]:5.2f}, {body_pos[1]:5.2f}, {body_pos[2]:5.2f}) | "
                          f"L-Grip: ({left_pos[0]:5.2f}, {left_pos[1]:5.2f}, {left_pos[2]:5.2f})")
                
                # Check duration
                if duration and (time.time() - start_time) >= duration:
                    break
                
                step_count += 1
                
                # Real-time sync
                time_until_next = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next > 0:
                    time.sleep(time_until_next)
        
        print("\n👋 Visualization ended")
    
    def _render_overlays(self, viewer):
        """Legacy method - replaced by _render_visualization_geoms"""
        pass


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    """Main function for Phase 1: Workspace visualization"""
    
    print("\n" + "="*70)
    print("MUJOCO WALL-CRAWLER SIMULATION - PHASE 1")
    print("Workspace Mapping & Path Planning Visualization")
    print("="*70)
    
    try:
        # Create simulation
        sim = WallCrawlerMuJoCoSimulation(model_path="dual_arm_robot.xml")
        
        # Define start and goal positions
        # ISS Module dimensions: X(-2.2 to 4.0), Y(-0.6 to 1.8), Z(0 to 2.0)
        # Start: On the FRONT wall (y=1.8), middle height - robot body stays inside
        start_pos = (1.0, 1.8, 1.0)  # Front wall, center height
        
        # Goal: On the BACK wall (y=-0.6), different X position
        goal_pos = (2.0, -0.6, 1.0)   # Back wall, center height
        
        # Set start and goal (this triggers path planning)
        sim.set_start_and_goal(start_pos, goal_pos)
        
        # Run visualization
        print("\nStarting MuJoCo visualization...")
        print("The robot will be displayed with the planned path.")
        print("(Arm control will be implemented in Phase 2)")
        
        sim.run_visualization()
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Make sure dual_arm_robot.xml is in the current directory.")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
