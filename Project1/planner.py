
import numpy as np
import heapq
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass, field
from enums import CrawlerState, RobotState

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

# KUKA iiwa14 workspace
KUKA_REACH = 1.25  # meters (including gripper)

@dataclass
class CrawlerState_Data:
    """State representation for the wall-crawler robot position (Private to Planner)"""
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

# Map the enum usage to this dataclass for compatibility with existing logic
CrawlerState = CrawlerState_Data

@dataclass(order=True)
class PriorityNode:
    """Node for A* priority queue"""
    f_cost: float
    state: CrawlerState = field(compare=False)
    g_cost: float = field(compare=False)
    parent: Optional['PriorityNode'] = field(compare=False, default=None)

class WallPositionMapper:
    """
    Maps discrete positions on the ISS module walls where the robot can grip.
    """
    
    def __init__(self, step_size: float = 0.4, arm_reach: float = KUKA_REACH):
        self.step_size = step_size
        self.arm_reach = arm_reach
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

class PathPlanner:
    """A* path planner for wall-crawling robot"""
    
    def __init__(self, wall_mapper: WallPositionMapper):
        self.mapper = wall_mapper
    
    def heuristic(self, state: CrawlerState, goal_pos: Tuple[float, float, float]) -> float:
        """Estimate cost to reach goal from state."""
        pos = state.position
        dist = np.linalg.norm(np.array(pos) - np.array(goal_pos))
        return dist
    
    def find_path(self, start_pos: Tuple[float, float, float],
                  goal_pos: Tuple[float, float, float],
                  start_arm: str = 'left') -> Optional[List[CrawlerState]]:
        """
        Find path from start to goal using A* algorithm.
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
