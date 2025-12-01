#!/usr/bin/env python3
"""
3D Dual-Arm Wall-Crawler Robot - ISS Module Environment

This script simulates a dual-arm wall-crawling robot in a 3D environment
matching the ISS module dimensions from the dual-arm simulation.

Environment dimensions (from dual_arm_robot.xml):
- X: -2.2m to 4.0m (6.2m wide)
- Y: -0.6m to 1.8m (2.4m deep)  
- Z: 0m to 2.0m (2.0m high)

Dual-Arm Configuration (from dual_arm_robot.xml):
- Left arm base: X = -0.25m (relative to central body)
- Right arm base: X = +0.25m (relative to central body)
- Arm separation: 0.5m
- Each arm reach: 0.82m (KUKA iiwa14)

The robot uses two KUKA arms as "feet":
- LEFT ARM (blue) - grips the wall
- RIGHT ARM (orange) - grips the wall
One arm stays fixed while the other moves to the next position.
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import heapq
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set, Dict
import time
import imageio.v2 as imageio
import tempfile
import os


# ============================================================================
# ENVIRONMENT CONFIGURATION - Based on dual_arm_robot.xml
# ============================================================================

# ISS Module dimensions (extracted from dual_arm_robot.xml)
ISS_MODULE = {
    'x_min': -2.2,   # wall_x_neg position
    'x_max': 4.0,    # wall_x_pos position
    'y_min': -0.6,   # wall_y_neg position
    'y_max': 1.8,    # wall_y_pos position
    'z_min': 0.0,    # floor level
    'z_max': 2.0,    # ceiling level
}

# Derived dimensions
ISS_WIDTH = ISS_MODULE['x_max'] - ISS_MODULE['x_min']   # 6.2m
ISS_DEPTH = ISS_MODULE['y_max'] - ISS_MODULE['y_min']   # 2.4m
ISS_HEIGHT = ISS_MODULE['z_max'] - ISS_MODULE['z_min']  # 2.0m

# DUAL ARM CONFIGURATION (from dual_arm_robot.xml)
LEFT_ARM_OFFSET = -0.25   # X offset from central body
RIGHT_ARM_OFFSET = 0.25   # X offset from central body
ARM_SEPARATION = RIGHT_ARM_OFFSET - LEFT_ARM_OFFSET  # 0.5m

# KUKA iiwa14 workspace - approximately 0.82m reach
KUKA_REACH = 0.82  # meters


@dataclass
class CrawlerState3D:
    """State representation for the dual-arm wall-crawler robot"""
    position: Tuple[float, float, float]  # Central body position on wall
    wall: str  # Which wall the robot is on
    active_arm: str  # Which arm is currently gripping: 'left' or 'right'
    
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
    state: CrawlerState3D = field(compare=False)
    g_cost: float = field(compare=False)
    parent: Optional['PriorityNode'] = field(compare=False, default=None)


class ISSModuleWallCrawler:
    """
    Dual-arm wall-crawling robot in ISS module environment.
    
    Uses two KUKA iiwa14 arms with 0.5m separation.
    Each arm has 0.82m reach.
    """
    
    def __init__(self, step_size: float = 0.4, arm_reach: float = KUKA_REACH, 
                 arm_separation: float = ARM_SEPARATION):
        """
        Initialize the ISS module dual-arm wall crawler.
        
        Args:
            step_size: Distance between discrete positions on walls
            arm_reach: Maximum reach of each arm (KUKA iiwa14)
            arm_separation: Distance between the two arms
        """
        # Store ISS module dimensions
        self.x_min = ISS_MODULE['x_min']
        self.x_max = ISS_MODULE['x_max']
        self.y_min = ISS_MODULE['y_min']
        self.y_max = ISS_MODULE['y_max']
        self.z_min = ISS_MODULE['z_min']
        self.z_max = ISS_MODULE['z_max']
        
        self.width = ISS_WIDTH
        self.depth = ISS_DEPTH
        self.height = ISS_HEIGHT
        
        self.step_size = step_size
        self.arm_reach = arm_reach
        self.arm_separation = arm_separation  # Distance between arms
        
        # Create wall positions
        self.wall_positions = self._create_wall_positions()
        
        print(f"ISS Module Environment:")
        print(f"  X: {self.x_min}m to {self.x_max}m ({self.width}m wide)")
        print(f"  Y: {self.y_min}m to {self.y_max}m ({self.depth}m deep)")
        print(f"  Z: {self.z_min}m to {self.z_max}m ({self.height}m high)")
        print(f"\nDual-Arm Configuration:")
        print(f"  Left arm offset: {LEFT_ARM_OFFSET}m")
        print(f"  Right arm offset: {RIGHT_ARM_OFFSET}m")
        print(f"  Arm separation: {self.arm_separation}m")
        print(f"  Each arm reach (KUKA iiwa14): {self.arm_reach}m")
        print(f"\n  Step size: {self.step_size}m")
        print(f"  Total wall positions: {len(self.wall_positions)}")
        
    def _create_wall_positions(self) -> List[Tuple[Tuple[float, float, float], str]]:
        """Create discrete positions on all 6 walls"""
        positions = []
        
        # Generate grid points
        xs = np.arange(self.x_min, self.x_max + self.step_size/2, self.step_size)
        ys = np.arange(self.y_min, self.y_max + self.step_size/2, self.step_size)
        zs = np.arange(self.z_min, self.z_max + self.step_size/2, self.step_size)
        
        # Floor (z = z_min)
        for x in xs:
            for y in ys:
                positions.append(((x, y, self.z_min), 'floor'))
        
        # Ceiling (z = z_max)
        for x in xs:
            for y in ys:
                positions.append(((x, y, self.z_max), 'ceiling'))
        
        # Front wall (y = y_min)
        for x in xs:
            for z in zs[1:-1]:
                positions.append(((x, self.y_min, z), 'front'))
        
        # Back wall (y = y_max)
        for x in xs:
            for z in zs[1:-1]:
                positions.append(((x, self.y_max, z), 'back'))
        
        # Left wall (x = x_min)
        for y in ys[1:-1]:
            for z in zs[1:-1]:
                positions.append(((self.x_min, y, z), 'left'))
        
        # Right wall (x = x_max)
        for y in ys[1:-1]:
            for z in zs[1:-1]:
                positions.append(((self.x_max, y, z), 'right'))
        
        return positions
    
    def get_distance(self, pos1, pos2) -> float:
        """Euclidean distance between two 3D positions"""
        return np.sqrt((pos1[0] - pos2[0])**2 + 
                      (pos1[1] - pos2[1])**2 + 
                      (pos1[2] - pos2[2])**2)
    
    def get_wall_normal(self, wall: str) -> Tuple[float, float, float]:
        """Get inward-pointing normal for a wall"""
        normals = {
            'floor': (0, 0, 1),
            'ceiling': (0, 0, -1),
            'front': (0, 1, 0),
            'back': (0, -1, 0),
            'left': (1, 0, 0),
            'right': (-1, 0, 0)
        }
        return normals.get(wall, (0, 0, 1))
    
    def can_transition(self, wall1: str, wall2: str) -> bool:
        """Check if robot can transition between walls"""
        adjacent = {
            'floor': ['front', 'back', 'left', 'right'],
            'ceiling': ['front', 'back', 'left', 'right'],
            'front': ['floor', 'ceiling', 'left', 'right'],
            'back': ['floor', 'ceiling', 'left', 'right'],
            'left': ['floor', 'ceiling', 'front', 'back'],
            'right': ['floor', 'ceiling', 'front', 'back']
        }
        return wall2 in adjacent.get(wall1, []) or wall1 == wall2
    
    def get_neighbors(self, state: CrawlerState3D) -> List[CrawlerState3D]:
        """Get reachable neighbor states for dual-arm robot"""
        neighbors = []
        current_pos = state.position
        
        for target_pos, target_wall in self.wall_positions:
            if target_pos == current_pos:
                continue
            
            dist = self.get_distance(current_pos, target_pos)
            if dist > self.arm_reach:
                continue
            
            if not self.can_transition(state.wall, target_wall):
                continue
            
            # Switch which arm is gripping
            new_active_arm = 'right' if state.active_arm == 'left' else 'left'
            neighbors.append(CrawlerState3D(target_pos, target_wall, new_active_arm))
        
        return neighbors
    
    def heuristic(self, pos1, pos2) -> float:
        """Heuristic: Euclidean distance"""
        return self.get_distance(pos1, pos2)
    
    def find_nearest_wall_position(self, pos):
        """Find nearest valid wall position"""
        min_dist = float('inf')
        nearest = self.wall_positions[0]
        
        for wall_pos, wall in self.wall_positions:
            dist = self.get_distance(pos, wall_pos)
            if dist < min_dist:
                min_dist = dist
                nearest = (wall_pos, wall)
        
        return nearest
    
    def find_path_astar(self, start_pos, goal_pos, start_arm: str = 'left'):
        """Find optimal path using A*"""
        start_wall_pos, start_wall = self.find_nearest_wall_position(start_pos)
        goal_wall_pos, goal_wall = self.find_nearest_wall_position(goal_pos)
        
        print(f"Start: {start_wall_pos} on {start_wall}")
        print(f"Goal: {goal_wall_pos} on {goal_wall}")
        
        start_state = CrawlerState3D(start_wall_pos, start_wall, start_arm)
        
        open_set = []
        start_node = PriorityNode(
            f_cost=self.heuristic(start_wall_pos, goal_wall_pos),
            state=start_state,
            g_cost=0.0,
            parent=None
        )
        heapq.heappush(open_set, start_node)
        
        closed_set: Set[CrawlerState3D] = set()
        g_costs: Dict[CrawlerState3D, float] = {start_state: 0.0}
        
        iterations = 0
        max_iterations = 100000
        
        while open_set and iterations < max_iterations:
            iterations += 1
            current_node = heapq.heappop(open_set)
            current_state = current_node.state
            
            if self.get_distance(current_state.position, goal_wall_pos) < 0.01:
                path = []
                node = current_node
                while node is not None:
                    path.append(node.state)
                    node = node.parent
                path.reverse()
                print(f"A* found path: {len(path)} steps in {iterations} iterations")
                return path
            
            if current_state in closed_set:
                continue
            closed_set.add(current_state)
            
            for next_state in self.get_neighbors(current_state):
                if next_state in closed_set:
                    continue
                
                move_cost = self.get_distance(current_state.position, next_state.position)
                tentative_g = current_node.g_cost + move_cost
                
                if next_state in g_costs and tentative_g >= g_costs[next_state]:
                    continue
                
                g_costs[next_state] = tentative_g
                f_cost = tentative_g + self.heuristic(next_state.position, goal_wall_pos)
                
                next_node = PriorityNode(
                    f_cost=f_cost,
                    state=next_state,
                    g_cost=tentative_g,
                    parent=current_node
                )
                heapq.heappush(open_set, next_node)
        
        print(f"A* failed after {iterations} iterations")
        return None


class ISSModuleVisualizer:
    """Visualizer for ISS module dual-arm wall crawler"""
    
    def __init__(self, planner: ISSModuleWallCrawler):
        self.planner = planner
        self.fig = None
        self.ax = None
        
        self.colors = {
            'floor': '#D4E6F1',
            'ceiling': '#FCF3CF',
            'walls': '#E8DAEF',
            'body': '#2C3E50',           # Dark gray central body
            'left_arm': '#3498DB',        # BLUE for left arm
            'right_arm': '#E67E22',       # ORANGE for right arm
            'left_workspace': '#5DADE2',  # Light blue for left arm workspace
            'right_workspace': '#F39C12', # Gold/orange for right arm workspace
            'path': '#1ABC9C',
            'start': '#2ECC71',
            'goal': '#E74C3C',
        }
    
    def setup_plot(self, title="ISS Module Wall-Crawler"):
        """Setup 3D plot"""
        self.fig = plt.figure(figsize=(14, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        margin = 0.5
        self.ax.set_xlim(self.planner.x_min - margin, self.planner.x_max + margin)
        self.ax.set_ylim(self.planner.y_min - margin, self.planner.y_max + margin)
        self.ax.set_zlim(self.planner.z_min - margin, self.planner.z_max + margin)
        
        self.ax.set_xlabel('X (meters)', fontsize=10)
        self.ax.set_ylabel('Y (meters)', fontsize=10)
        self.ax.set_zlabel('Z (meters)', fontsize=10)
        self.ax.set_title(title, fontsize=14, fontweight='bold')
        
        self.ax.view_init(elev=20, azim=45)
    
    def draw_container(self, alpha=0.2):
        """Draw ISS module walls"""
        x_min, x_max = self.planner.x_min, self.planner.x_max
        y_min, y_max = self.planner.y_min, self.planner.y_max
        z_min, z_max = self.planner.z_min, self.planner.z_max
        
        faces = [
            # Floor
            [[x_min, y_min, z_min], [x_max, y_min, z_min], 
             [x_max, y_max, z_min], [x_min, y_max, z_min]],
            # Ceiling
            [[x_min, y_min, z_max], [x_max, y_min, z_max], 
             [x_max, y_max, z_max], [x_min, y_max, z_max]],
            # Front (y_min)
            [[x_min, y_min, z_min], [x_max, y_min, z_min], 
             [x_max, y_min, z_max], [x_min, y_min, z_max]],
            # Back (y_max)
            [[x_min, y_max, z_min], [x_max, y_max, z_min], 
             [x_max, y_max, z_max], [x_min, y_max, z_max]],
            # Left (x_min)
            [[x_min, y_min, z_min], [x_min, y_max, z_min], 
             [x_min, y_max, z_max], [x_min, y_min, z_max]],
            # Right (x_max)
            [[x_max, y_min, z_min], [x_max, y_max, z_min], 
             [x_max, y_max, z_max], [x_max, y_min, z_max]],
        ]
        
        face_colors = [
            self.colors['floor'], self.colors['ceiling'],
            self.colors['walls'], self.colors['walls'],
            self.colors['walls'], self.colors['walls'],
        ]
        
        for face, color in zip(faces, face_colors):
            poly = Poly3DCollection([face], alpha=alpha, facecolor=color, 
                                   edgecolor='black', linewidth=1)
            self.ax.add_collection3d(poly)
        
        # Draw grid points (batched for performance)
        all_positions = np.array([pos for (pos, wall) in self.planner.wall_positions])
        self.ax.scatter(all_positions[:, 0], all_positions[:, 1], all_positions[:, 2], 
                       c='gray', s=3, alpha=0.2)
    
    def draw_workspace_sphere(self, center, radius, color='gold', alpha=0.15):
        """Draw a sphere showing the KUKA arm workspace reach.
        Returns the surface object so it can be removed later."""
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 15)
        
        x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
        y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
        z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
        
        surface = self.ax.plot_surface(x, y, z, color=color, alpha=alpha, linewidth=0)
        return surface
    
    def draw_path_line(self, path: List[CrawlerState3D]):
        """Draw path"""
        if not path:
            return
        
        xs = [s.position[0] for s in path]
        ys = [s.position[1] for s in path]
        zs = [s.position[2] for s in path]
        
        self.ax.plot(xs, ys, zs, color=self.colors['path'], linewidth=2, alpha=0.7)
        self.ax.scatter([xs[0]], [ys[0]], [zs[0]], c=self.colors['start'], 
                       s=200, marker='*', label='Start')
        self.ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], c=self.colors['goal'], 
                       s=200, marker='*', label='Goal')
    
    def draw_robot(self, state: CrawlerState3D, prev_state=None, show_workspace=True):
        """Draw dual-arm robot with separate workspace for each arm"""
        pos = np.array(state.position)
        wall = state.wall
        active_arm = state.active_arm  # 'left' or 'right'
        normal = np.array(self.planner.get_wall_normal(wall))
        
        # Get tangent direction based on wall
        if wall in ['floor', 'ceiling']:
            tangent1 = np.array([1, 0, 0])
        elif wall in ['front', 'back']:
            tangent1 = np.array([1, 0, 0])
        else:
            tangent1 = np.array([0, 1, 0])
        
        # Calculate arm positions based on arm separation
        # Arms are offset from body center by +/- half the separation
        arm_offset = self.planner.arm_separation / 2  # 0.25m
        
        if prev_state is not None:
            prev_pos = np.array(prev_state.position)
            
            # The active_arm just moved to the current position
            # The other arm is still at the previous position
            if active_arm == 'left':
                # Left arm just moved to current position
                left_arm_pos = pos
                right_arm_pos = prev_pos
            else:
                # Right arm just moved to current position
                left_arm_pos = prev_pos
                right_arm_pos = pos
        else:
            # Initial state - both arms at starting positions
            left_arm_pos = pos - tangent1 * arm_offset
            right_arm_pos = pos + tangent1 * arm_offset
        
        # Body position (between the two arms, offset from wall)
        body_pos = (left_arm_pos + right_arm_pos) / 2 + normal * 0.2
        
        elements = []
        
        # Draw workspace spheres for EACH arm (track them for removal)
        if show_workspace:
            # Left arm workspace (BLUE)
            left_ws = self.draw_workspace_sphere(left_arm_pos, self.planner.arm_reach, 
                                                  color=self.colors['left_workspace'], alpha=0.08)
            # Right arm workspace (ORANGE)
            right_ws = self.draw_workspace_sphere(right_arm_pos, self.planner.arm_reach, 
                                                   color=self.colors['right_workspace'], alpha=0.08)
            elements.extend([left_ws, right_ws])
        
        # Draw arms (end-effector positions) - consistent colors!
        # LEFT ARM is always BLUE
        left_arm = self.ax.scatter(*left_arm_pos, c=self.colors['left_arm'], 
                                   s=180, marker='o', edgecolor='darkblue', linewidth=2, 
                                   zorder=10, label='Left Arm' if not hasattr(self, '_legend_set') else '')
        # RIGHT ARM is always ORANGE
        right_arm = self.ax.scatter(*right_arm_pos, c=self.colors['right_arm'], 
                                    s=180, marker='o', edgecolor='darkorange', linewidth=2, 
                                    zorder=10, label='Right Arm' if not hasattr(self, '_legend_set') else '')
        elements.extend([left_arm, right_arm])
        self._legend_set = True
        
        # Draw body (central box)
        body = self.ax.scatter(*body_pos, c=self.colors['body'], s=300, 
                               marker='s', edgecolor='black', linewidth=2, zorder=15)
        elements.append(body)
        
        # Draw arms (legs connecting body to end-effectors)
        # Left arm link (BLUE)
        leg_left, = self.ax.plot([body_pos[0], left_arm_pos[0]], 
                                 [body_pos[1], left_arm_pos[1]],
                                 [body_pos[2], left_arm_pos[2]],
                                 color=self.colors['left_arm'], linewidth=6, zorder=5)
        # Right arm link (ORANGE)
        leg_right, = self.ax.plot([body_pos[0], right_arm_pos[0]], 
                                  [body_pos[1], right_arm_pos[1]],
                                  [body_pos[2], right_arm_pos[2]],
                                  color=self.colors['right_arm'], linewidth=6, zorder=5)
        elements.extend([leg_left, leg_right])
        
        return elements
    
    def animate_path(self, path: List[CrawlerState3D], save_file: str = None,
                     interval: float = 0.4):
        """Animate the path (skip static, go straight to animation)"""
        if not path:
            print("No path to animate!")
            return
        
        # Save GIF first
        if save_file:
            print(f"Saving animation to {save_file}...")
            self._save_animation_gif(path, save_file)
            print(f"GIF saved to {save_file}")
        
        # Show interactive animation
        print("Showing interactive animation...")
        self.setup_plot(f"ISS Module Dual-Arm Wall-Crawler | Arm Reach: {self.planner.arm_reach}m | Separation: {self.planner.arm_separation}m")
        self.draw_container(alpha=0.15)
        self.draw_path_line(path)
        
        plt.ion()
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.3)
        
        robot_elements = []
        
        try:
            for frame in range(len(path)):
                for elem in robot_elements:
                    try:
                        elem.remove()
                    except:
                        pass
                robot_elements = []
                
                state = path[frame]
                prev_state = path[frame - 1] if frame > 0 else None
                
                elements = self.draw_robot(state, prev_state, show_workspace=True)
                robot_elements.extend(elements)
                
                arm_name = "Left" if state.active_arm == 'left' else "Right"
                pos = state.position
                self.ax.set_title(
                    f"ISS Module Dual-Arm Wall-Crawler | Step {frame + 1}/{len(path)}\n"
                    f"Wall: {state.wall.capitalize()} | Holding: {arm_name} Arm | "
                    f"Pos: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) | "
                    f"Arm Reach: {self.planner.arm_reach}m",
                    fontsize=11, fontweight='bold'
                )
                
                self.fig.canvas.draw()
                self.fig.canvas.flush_events()
                plt.pause(interval)
            
            print("Animation complete!")
        except Exception as e:
            print(f"Animation interrupted: {e}")
        finally:
            plt.ioff()
            try:
                plt.show()
            except:
                pass
    
    def _save_animation_gif(self, path: List[CrawlerState3D], save_file: str):
        """Save animation as GIF with dual-arm visualization"""
        frames = []
        temp_dir = tempfile.mkdtemp()
        
        try:
            for frame in range(len(path)):
                fig = plt.figure(figsize=(10, 8))
                ax = fig.add_subplot(111, projection='3d')
                
                margin = 0.3
                ax.set_xlim(self.planner.x_min - margin, self.planner.x_max + margin)
                ax.set_ylim(self.planner.y_min - margin, self.planner.y_max + margin)
                ax.set_zlim(self.planner.z_min - margin, self.planner.z_max + margin)
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')
                ax.set_zlabel('Z (m)')
                
                # Draw container faces
                x_min, x_max = self.planner.x_min, self.planner.x_max
                y_min, y_max = self.planner.y_min, self.planner.y_max
                z_min, z_max = self.planner.z_min, self.planner.z_max
                
                faces = [
                    [[x_min, y_min, z_min], [x_max, y_min, z_min], 
                     [x_max, y_max, z_min], [x_min, y_max, z_min]],
                    [[x_min, y_min, z_max], [x_max, y_min, z_max], 
                     [x_max, y_max, z_max], [x_min, y_max, z_max]],
                    [[x_min, y_min, z_min], [x_max, y_min, z_min], 
                     [x_max, y_min, z_max], [x_min, y_min, z_max]],
                    [[x_min, y_max, z_min], [x_max, y_max, z_min], 
                     [x_max, y_max, z_max], [x_min, y_max, z_max]],
                    [[x_min, y_min, z_min], [x_min, y_max, z_min], 
                     [x_min, y_max, z_max], [x_min, y_min, z_max]],
                    [[x_max, y_min, z_min], [x_max, y_max, z_min], 
                     [x_max, y_max, z_max], [x_max, y_min, z_max]],
                ]
                for face in faces:
                    poly = Poly3DCollection([face], alpha=0.12, facecolor='lightblue', 
                                           edgecolor='black', linewidth=0.5)
                    ax.add_collection3d(poly)
                
                # Draw path
                xs = [s.position[0] for s in path]
                ys = [s.position[1] for s in path]
                zs = [s.position[2] for s in path]
                ax.plot(xs, ys, zs, color='blue', linewidth=2, alpha=0.5)
                ax.scatter([xs[0]], [ys[0]], [zs[0]], c='green', s=100, marker='*')
                ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], c='red', s=100, marker='*')
                
                # Draw dual-arm robot
                state = path[frame]
                prev_state = path[frame - 1] if frame > 0 else None
                pos = np.array(state.position)
                wall = state.wall
                active_arm = state.active_arm  # 'left' or 'right'
                
                normal = np.array(self.planner.get_wall_normal(wall))
                
                if wall in ['floor', 'ceiling']:
                    tangent1 = np.array([1, 0, 0])
                elif wall in ['front', 'back']:
                    tangent1 = np.array([1, 0, 0])
                else:
                    tangent1 = np.array([0, 1, 0])
                
                arm_offset = self.planner.arm_separation / 2
                
                if prev_state is not None:
                    prev_pos = np.array(prev_state.position)
                    if active_arm == 'left':
                        left_arm_pos = pos
                        right_arm_pos = prev_pos
                    else:
                        left_arm_pos = prev_pos
                        right_arm_pos = pos
                else:
                    left_arm_pos = pos - tangent1 * arm_offset
                    right_arm_pos = pos + tangent1 * arm_offset
                
                body_pos = (left_arm_pos + right_arm_pos) / 2 + normal * 0.15
                
                # Draw workspace spheres for each arm
                u = np.linspace(0, 2 * np.pi, 15)
                v = np.linspace(0, np.pi, 10)
                
                # Left arm workspace (BLUE)
                sphere_x = left_arm_pos[0] + self.planner.arm_reach * np.outer(np.cos(u), np.sin(v))
                sphere_y = left_arm_pos[1] + self.planner.arm_reach * np.outer(np.sin(u), np.sin(v))
                sphere_z = left_arm_pos[2] + self.planner.arm_reach * np.outer(np.ones(np.size(u)), np.cos(v))
                ax.plot_surface(sphere_x, sphere_y, sphere_z, color=self.colors['left_workspace'], alpha=0.08, linewidth=0)
                
                # Right arm workspace (ORANGE)
                sphere_x = right_arm_pos[0] + self.planner.arm_reach * np.outer(np.cos(u), np.sin(v))
                sphere_y = right_arm_pos[1] + self.planner.arm_reach * np.outer(np.sin(u), np.sin(v))
                sphere_z = right_arm_pos[2] + self.planner.arm_reach * np.outer(np.ones(np.size(u)), np.cos(v))
                ax.plot_surface(sphere_x, sphere_y, sphere_z, color=self.colors['right_workspace'], alpha=0.08, linewidth=0)
                
                # Draw robot parts with dual-arm colors
                # Left arm (BLUE)
                ax.scatter(*left_arm_pos, c=self.colors['left_arm'], s=120, marker='o', 
                          edgecolor='darkblue', linewidth=2)
                # Right arm (ORANGE)
                ax.scatter(*right_arm_pos, c=self.colors['right_arm'], s=120, marker='o', 
                          edgecolor='darkorange', linewidth=2)
                # Body
                ax.scatter(*body_pos, c=self.colors['body'], s=200, marker='s', edgecolor='black')
                
                # Arm links
                ax.plot([body_pos[0], left_arm_pos[0]], [body_pos[1], left_arm_pos[1]], 
                       [body_pos[2], left_arm_pos[2]], color=self.colors['left_arm'], linewidth=5)
                ax.plot([body_pos[0], right_arm_pos[0]], [body_pos[1], right_arm_pos[1]], 
                       [body_pos[2], right_arm_pos[2]], color=self.colors['right_arm'], linewidth=5)
                
                arm_name = "Left" if active_arm == 'left' else "Right"
                ax.set_title(f"ISS Module Dual-Arm Wall-Crawler | Step {frame + 1}/{len(path)}\n"
                            f"Wall: {wall.capitalize()} | Holding: {arm_name} Arm | "
                            f"Arm Reach: {self.planner.arm_reach}m | Separation: {self.planner.arm_separation}m",
                            fontsize=10, fontweight='bold')
                
                ax.view_init(elev=20, azim=45)
                
                frame_path = os.path.join(temp_dir, f'frame_{frame:03d}.png')
                fig.savefig(frame_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                
                frames.append(imageio.imread(frame_path))
            
            imageio.mimsave(save_file, frames, duration=0.4)
            
        finally:
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    """Main function"""
    print("=" * 70)
    print("ISS MODULE DUAL-ARM WALL-CRAWLER WITH KUKA ARM WORKSPACE")
    print("=" * 70)
    
    # Create planner with ISS module dimensions and KUKA reach
    planner = ISSModuleWallCrawler(
        step_size=0.5,  # Larger step for bigger environment
        arm_reach=KUKA_REACH,  # 0.82m KUKA arm reach
        arm_separation=0.5  # 0.5m between arms (from XML: left at -0.25, right at +0.25)
    )
    
    print("\n" + "=" * 70)
    print("ENVIRONMENT SUMMARY:")
    print(f"  Box dimensions: {ISS_WIDTH:.1f}m × {ISS_DEPTH:.1f}m × {ISS_HEIGHT:.1f}m")
    print(f"  KUKA iiwa14 arm reach: {KUKA_REACH}m")
    print(f"  Arm separation: {planner.arm_separation}m")
    print(f"  Reach-to-Width ratio: {KUKA_REACH / ISS_WIDTH:.2%}")
    print(f"  Reach-to-Depth ratio: {KUKA_REACH / ISS_DEPTH:.2%}")
    print(f"  Reach-to-Height ratio: {KUKA_REACH / ISS_HEIGHT:.2%}")
    print("=" * 70)
    
    # Start on floor near one corner
    start = (planner.x_min + 0.5, planner.y_min + 0.5, planner.z_min)
    
    # Random goal on a different wall
    np.random.seed(int(time.time() * 1000) % 2**32)
    valid_goals = [(pos, wall) for pos, wall in planner.wall_positions 
                   if planner.get_distance(pos, start) > 2.0]
    
    if valid_goals:
        goal_pos, _ = valid_goals[np.random.randint(len(valid_goals))]
        goal = goal_pos
    else:
        goal = (planner.x_max - 0.5, planner.y_max - 0.5, planner.z_max)
    
    print(f"\nStart: ({start[0]:.2f}, {start[1]:.2f}, {start[2]:.2f})")
    print(f"Goal: ({goal[0]:.2f}, {goal[1]:.2f}, {goal[2]:.2f})")
    
    # Find path
    print("\nRunning A* search...")
    start_time = time.time()
    path = planner.find_path_astar(start, goal)
    elapsed = time.time() - start_time
    
    if path:
        print(f"\nPath found in {elapsed:.3f}s with {len(path)} steps")
        for i, state in enumerate(path):
            arm = "Left Arm" if state.active_arm == 'left' else "Right Arm"
            pos = state.position
            print(f"  Step {i+1:2d}: {state.wall:7s} | "
                  f"({pos[0]:5.2f}, {pos[1]:5.2f}, {pos[2]:5.2f}) | Holding: {arm}")
    else:
        print("No path found!")
        return
    
    # Create visualizer and animate (skip static)
    viz = ISSModuleVisualizer(planner)
    
    print("\nGenerating animation (skipping static plot)...")
    viz.animate_path(path, save_file='wall_crawler_iss_module.gif', interval=0.4)


if __name__ == "__main__":
    main()
