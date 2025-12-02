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


# ============================================================================
# ENVIRONMENT CONFIGURATION - Based on dual_arm_robot.xml
# ============================================================================

ISS_MODULE = {
    'x_min': -2.2,   # wall_x_neg position
    'x_max': 4.0,    # wall_x_pos position
    'y_min': -0.6,   # wall_y_neg position
    'y_max': 1.8,    # wall_y_pos position
    'z_min': 0.0,    # floor level
    'z_max': 2.0,    # ceiling level
}

ISS_WIDTH = ISS_MODULE['x_max'] - ISS_MODULE['x_min']   # 6.2m
ISS_DEPTH = ISS_MODULE['y_max'] - ISS_MODULE['y_min']   # 2.4m
ISS_HEIGHT = ISS_MODULE['z_max'] - ISS_MODULE['z_min']  # 2.0m

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
        
        # Floor (z = z_min)
        for x in np.arange(self.x_min + margin, self.x_max - margin, self.step_size):
            for y in np.arange(self.y_min + margin, self.y_max - margin, self.step_size):
                positions.append(((x, y, self.z_min), 'floor'))
        
        # Ceiling (z = z_max)
        for x in np.arange(self.x_min + margin, self.x_max - margin, self.step_size):
            for y in np.arange(self.y_min + margin, self.y_max - margin, self.step_size):
                positions.append(((x, y, self.z_max), 'ceiling'))
        
        # Front wall (y = y_max)
        for x in np.arange(self.x_min + margin, self.x_max - margin, self.step_size):
            for z in np.arange(self.z_min + margin, self.z_max - margin, self.step_size):
                positions.append(((x, self.y_max, z), 'front'))
        
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
        tolerance = 0.05
        
        if abs(z - self.z_min) < tolerance:
            return 'floor'
        elif abs(z - self.z_max) < tolerance:
            return 'ceiling'
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
        
        # Initialize wall position mapper
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
    
    def set_start_and_goal(self, start: Tuple[float, float, float],
                           goal: Tuple[float, float, float]):
        """Set start and goal positions for path planning"""
        self.start_position = start
        self.goal_position = goal
        
        print(f"\n{'='*60}")
        print("PATH PLANNING")
        print(f"{'='*60}")
        print(f"Start: {start}")
        print(f"Goal: {goal}")
        
        # Transition to PLANNING state
        self._transition_state(RobotState.PLANNING)
        
        # Compute path
        self.current_path = self.path_planner.find_path(start, goal, start_arm='left')
        
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
            
            # Set camera
            viewer.cam.lookat[:] = [0.5, 0.6, 1.0]
            viewer.cam.distance = 5.0
            viewer.cam.azimuth = 135
            viewer.cam.elevation = -25
            
            # Reset simulation
            mujoco.mj_resetData(self.model, self.data)
            
            # Set initial robot position (floating in space, near start)
            if self.start_position:
                start_x, start_y, start_z = self.start_position
                # Position body above the start position
                self.data.qpos[0] = start_x
                self.data.qpos[1] = start_y
                self.data.qpos[2] = start_z + 0.5  # Offset above floor
            else:
                self.data.qpos[0] = 0.0
                self.data.qpos[1] = 0.5
                self.data.qpos[2] = 0.5
            
            self.data.qpos[3] = 1.0   # quat w
            self.data.qpos[4:7] = 0.0 # quat xyz
            
            # Set home position for arms
            # Left arm
            self.data.qpos[7] = 0.0
            self.data.qpos[8] = 0.785398
            self.data.qpos[9] = 0.0
            self.data.qpos[10] = -1.5708
            self.data.qpos[11] = 0.0
            self.data.qpos[12] = 0.0
            self.data.qpos[13] = 0.0
            
            # Right arm
            self.data.qpos[22] = 0.0
            self.data.qpos[23] = 0.785398
            self.data.qpos[24] = 0.0
            self.data.qpos[25] = -1.5708
            self.data.qpos[26] = 0.0
            self.data.qpos[27] = 0.0
            self.data.qpos[28] = 0.0
            
            # Zero velocities to keep robot stable
            self.data.qvel[:] = 0.0
            
            mujoco.mj_forward(self.model, self.data)
            
            start_time = time.time()
            step_count = 0
            first_render = True
            
            print(f"\n📍 Visualizing {len(self.wall_mapper.wall_positions)} wall grip positions...")
            print(f"📍 Workspace spheres: Left (blue) and Right (orange) with {WORKSPACE_SPHERE_RADIUS}m radius (centered at elbow)")
            if self.current_path:
                print(f"📍 Showing path with {len(self.current_path)} waypoints")
            
            while viewer.is_running():
                step_start = time.time()
                
                # Step physics - robot is now FREE to move!
                mujoco.mj_step(self.model, self.data)
                
                # Render visualization geometries using viewer's user scene
                with viewer.lock():
                    self._render_visualization_geoms(viewer)
                    if first_render:
                        print(f"✓ Added {viewer.user_scn.ngeom} custom geometries to scene")
                        print("✓ Physics enabled - robot is free-floating!")
                        first_render = False
                
                # Sync viewer
                viewer.sync()
                
                # Print status periodically
                if step_count % 500 == 0 and step_count > 0:
                    left_pos = self.get_left_gripper_pos()
                    right_pos = self.get_right_gripper_pos()
                    body_pos = self.get_central_body_pos()
                    
                    print(f"[{self.robot_state.name:15s}] "
                          f"Body: ({body_pos[0]:5.2f}, {body_pos[1]:5.2f}, {body_pos[2]:5.2f}) | "
                          f"L-Grip: ({left_pos[0]:5.2f}, {left_pos[1]:5.2f}, {left_pos[2]:5.2f}) | "
                          f"R-Grip: ({right_pos[0]:5.2f}, {right_pos[1]:5.2f}, {right_pos[2]:5.2f})")
                
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
        # Start: On the floor near the center
        start_pos = (0.0, 0.5, 0.0)  # Floor
        
        # Goal: On the ceiling
        goal_pos = (1.0, 1.0, 2.0)   # Ceiling
        
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
