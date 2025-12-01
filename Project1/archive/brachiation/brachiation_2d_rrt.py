#!/usr/bin/env python3
"""
2D Brachiation Robot Simulation with RRT Path Planning

This script simulates a two-link robot (like two connected lines) that moves 
through a 2D environment in zero-gravity by alternating which end is anchored 
(brachiation-style movement). RRT is used to plan the sequence of movements.

The robot consists of:
- Two rigid links connected at a central pivot
- Can anchor/grasp at either endpoint
- Swings freely around the anchored point
- Goal: Move one endpoint to reach a target location

Author: Space Robotics Project
Date: 2024
"""

import os
import sys

# Check for headless mode before importing matplotlib
HEADLESS = '--headless' in sys.argv or os.environ.get('DISPLAY') is None

import matplotlib
if HEADLESS:
    matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.collections import LineCollection
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass
from typing import List, Tuple, Optional, Set
import random
import time


# ==============================================================================
# Robot Configuration
# ==============================================================================

@dataclass
class RobotState:
    """
    State of the two-link brachiation robot.
    
    The robot has two links connected at a pivot point.
    - anchor_pos: Position of the currently anchored end (x, y)
    - angle: Angle of the robot from anchor to free end (radians, 0 = right, CCW positive)
    - anchor_end: Which end is anchored ('A' or 'B')
    """
    anchor_pos: np.ndarray  # (x, y) position of anchored end
    angle: float            # Angle from anchor to pivot to free end
    anchor_end: str         # 'A' or 'B' - which end is currently anchored
    
    def __post_init__(self):
        if isinstance(self.anchor_pos, (list, tuple)):
            self.anchor_pos = np.array(self.anchor_pos, dtype=float)
    
    def copy(self):
        return RobotState(self.anchor_pos.copy(), self.angle, self.anchor_end)
    
    def __hash__(self):
        return hash((tuple(self.anchor_pos), self.angle, self.anchor_end))
    
    def __eq__(self, other):
        if not isinstance(other, RobotState):
            return False
        return (np.allclose(self.anchor_pos, other.anchor_pos) and 
                np.isclose(self.angle, other.angle) and 
                self.anchor_end == other.anchor_end)


class BrachiationRobot:
    """
    Two-link brachiation robot model.
    
    The robot looks like this:
    
        End A -------- Pivot -------- End B
              (link1)        (link2)
    
    Each link has length L. The robot can anchor at either End A or End B,
    and swing around that anchor point.
    """
    
    def __init__(self, link_length: float = 1.0):
        """
        Initialize the robot.
        
        Args:
            link_length: Length of each link (total span = 2 * link_length)
        """
        self.link_length = link_length
        self.total_length = 2 * link_length
    
    def get_pivot_position(self, state: RobotState) -> np.ndarray:
        """Get the position of the central pivot point."""
        # From anchor, move link_length in the direction of angle
        offset = self.link_length * np.array([np.cos(state.angle), np.sin(state.angle)])
        return state.anchor_pos + offset
    
    def get_free_end_position(self, state: RobotState) -> np.ndarray:
        """Get the position of the free (non-anchored) end."""
        pivot = self.get_pivot_position(state)
        # Free end is link_length further in the same direction
        offset = self.link_length * np.array([np.cos(state.angle), np.sin(state.angle)])
        return pivot + offset
    
    def get_all_positions(self, state: RobotState) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get positions of all three points: anchor, pivot, free end.
        
        Returns:
            (anchor_pos, pivot_pos, free_end_pos)
        """
        anchor = state.anchor_pos.copy()
        pivot = self.get_pivot_position(state)
        free_end = self.get_free_end_position(state)
        return anchor, pivot, free_end
    
    def get_end_positions(self, state: RobotState) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get positions of End A and End B.
        
        Returns:
            (end_a_pos, end_b_pos)
        """
        anchor, pivot, free_end = self.get_all_positions(state)
        if state.anchor_end == 'A':
            return anchor, free_end
        else:  # anchor_end == 'B'
            return free_end, anchor
    
    def swing_to_angle(self, state: RobotState, new_angle: float) -> RobotState:
        """
        Swing the robot to a new angle (keeping anchor fixed).
        
        Args:
            state: Current robot state
            new_angle: New angle in radians
            
        Returns:
            New robot state
        """
        return RobotState(state.anchor_pos.copy(), new_angle, state.anchor_end)
    
    def switch_anchor(self, state: RobotState) -> RobotState:
        """
        Switch which end is anchored (brachiation move).
        
        The free end becomes the new anchor, and the old anchor becomes free.
        """
        # Get current free end position - this becomes the new anchor
        new_anchor = self.get_free_end_position(state)
        
        # Calculate new angle (pointing back towards old anchor)
        old_anchor = state.anchor_pos
        direction = old_anchor - new_anchor
        new_angle = np.arctan2(direction[1], direction[0])
        
        # Switch anchor designation
        new_anchor_end = 'B' if state.anchor_end == 'A' else 'A'
        
        return RobotState(new_anchor, new_angle, new_anchor_end)


# ==============================================================================
# Environment Configuration
# ==============================================================================

@dataclass
class Obstacle:
    """Rectangular obstacle in the environment."""
    x: float
    y: float
    width: float
    height: float
    
    def contains_point(self, point: np.ndarray) -> bool:
        """Check if a point is inside the obstacle."""
        return (self.x <= point[0] <= self.x + self.width and
                self.y <= point[1] <= self.y + self.height)
    
    def intersects_segment(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        """Check if a line segment intersects the obstacle."""
        # Check if segment endpoints are inside
        if self.contains_point(p1) or self.contains_point(p2):
            return True
        
        # Check intersection with each edge
        corners = [
            (np.array([self.x, self.y]), np.array([self.x + self.width, self.y])),
            (np.array([self.x + self.width, self.y]), np.array([self.x + self.width, self.y + self.height])),
            (np.array([self.x + self.width, self.y + self.height]), np.array([self.x, self.y + self.height])),
            (np.array([self.x, self.y + self.height]), np.array([self.x, self.y]))
        ]
        
        for c1, c2 in corners:
            if self._segments_intersect(p1, p2, c1, c2):
                return True
        return False
    
    @staticmethod
    def _segments_intersect(p1, p2, p3, p4) -> bool:
        """Check if two line segments intersect."""
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        return (ccw(p1, p3, p4) != ccw(p2, p3, p4) and 
                ccw(p1, p2, p3) != ccw(p1, p2, p4))


class Environment:
    """2D environment with obstacles and boundaries."""
    
    def __init__(self, width: float = 20.0, height: float = 15.0):
        self.width = width
        self.height = height
        self.obstacles: List[Obstacle] = []
        self.hold_points: List[np.ndarray] = []  # Valid points where robot can anchor
        
    def add_obstacle(self, x: float, y: float, w: float, h: float):
        """Add a rectangular obstacle."""
        self.obstacles.append(Obstacle(x, y, w, h))
    
    def add_hold_point(self, x: float, y: float):
        """Add a point where the robot can anchor/grasp."""
        self.hold_points.append(np.array([x, y]))
    
    def is_point_valid(self, point: np.ndarray) -> bool:
        """Check if a point is within bounds and not inside obstacles."""
        # Check bounds
        if point[0] < 0 or point[0] > self.width:
            return False
        if point[1] < 0 or point[1] > self.height:
            return False
        
        # Check obstacles
        for obs in self.obstacles:
            if obs.contains_point(point):
                return False
        return True
    
    def is_segment_valid(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        """Check if a line segment doesn't intersect any obstacles."""
        for obs in self.obstacles:
            if obs.intersects_segment(p1, p2):
                return False
        return True
    
    def is_robot_state_valid(self, robot: BrachiationRobot, state: RobotState) -> bool:
        """Check if a robot state is collision-free."""
        anchor, pivot, free_end = robot.get_all_positions(state)
        
        # Check all points are valid
        if not all(self.is_point_valid(p) for p in [anchor, pivot, free_end]):
            return False
        
        # Check both links don't intersect obstacles
        if not self.is_segment_valid(anchor, pivot):
            return False
        if not self.is_segment_valid(pivot, free_end):
            return False
        
        return True
    
    def get_nearest_hold(self, point: np.ndarray) -> Optional[np.ndarray]:
        """Get the nearest hold point to the given point."""
        if not self.hold_points:
            return None
        distances = [np.linalg.norm(p - point) for p in self.hold_points]
        return self.hold_points[np.argmin(distances)]
    
    def create_default_environment(self):
        """Create a default environment with walls and hold points."""
        # Add some walls/obstacles (thinner walls for easier navigation)
        self.add_obstacle(6, 2, 0.5, 6)   # Vertical wall
        self.add_obstacle(12, 7, 0.5, 6)  # Another wall
        
        # Add hold points (places where robot can grasp)
        # Create a denser grid of hold points avoiding obstacles
        for x in np.arange(1, self.width - 1, 2.0):
            for y in np.arange(1, self.height - 1, 2.0):
                point = np.array([x, y])
                if self.is_point_valid(point):
                    # Check not too close to obstacles
                    valid = True
                    for obs in self.obstacles:
                        if (obs.x - 1.0 <= x <= obs.x + obs.width + 1.0 and
                            obs.y - 1.0 <= y <= obs.y + obs.height + 1.0):
                            valid = False
                            break
                    if valid:
                        self.add_hold_point(x, y)


# ==============================================================================
# RRT Path Planner
# ==============================================================================

@dataclass
class RRTNode:
    """Node in the RRT tree."""
    state: RobotState
    parent: Optional['RRTNode'] = None
    action: Optional[str] = None  # 'swing' or 'switch'
    
    def __hash__(self):
        return hash(self.state)


class BrachiationRRT:
    """
    RRT-based path planner for the brachiation robot.
    
    The configuration space consists of:
    - Anchor position (discrete - must be at hold points)
    - Angle (continuous)
    - Which end is anchored (discrete - A or B)
    
    Actions:
    - Swing: Change angle while keeping anchor fixed
    - Switch: Switch anchor to free end (requires free end to be at a hold point)
    """
    
    def __init__(self, robot: BrachiationRobot, env: Environment, 
                 goal_pos: np.ndarray, goal_tolerance: float = 0.5):
        self.robot = robot
        self.env = env
        self.goal_pos = goal_pos
        self.goal_tolerance = goal_tolerance
        
        # RRT parameters
        self.max_iterations = 3000
        self.swing_step = np.pi / 4  # Max angle change per swing step
        self.goal_bias = 0.4  # Probability of sampling towards goal
        self.hold_snap_distance = 1.8  # Distance to snap to hold points (larger for easier grasping)
        
        # Tree storage
        self.nodes: List[RRTNode] = []
    
    def distance_to_goal(self, state: RobotState) -> float:
        """Calculate minimum distance from either end to goal."""
        end_a, end_b = self.robot.get_end_positions(state)
        return min(np.linalg.norm(end_a - self.goal_pos),
                   np.linalg.norm(end_b - self.goal_pos))
    
    def is_goal_reached(self, state: RobotState) -> bool:
        """Check if either end has reached the goal."""
        return self.distance_to_goal(state) < self.goal_tolerance
    
    def sample_random_state(self) -> RobotState:
        """Sample a random valid state."""
        if random.random() < self.goal_bias:
            # Sample towards goal - find hold nearest to goal
            nearest_to_goal = min(self.env.hold_points, 
                                  key=lambda h: np.linalg.norm(h - self.goal_pos))
            direction = self.goal_pos - nearest_to_goal
            angle = np.arctan2(direction[1], direction[0])
            # Add some noise
            angle += random.uniform(-np.pi/4, np.pi/4)
            return RobotState(nearest_to_goal.copy(), angle, random.choice(['A', 'B']))
        
        # Random sample from hold points
        if self.env.hold_points:
            anchor = random.choice(self.env.hold_points).copy()
        else:
            anchor = np.array([5.0, 5.0])
        angle = random.uniform(-np.pi, np.pi)
        anchor_end = random.choice(['A', 'B'])
        return RobotState(anchor, angle, anchor_end)
    
    def find_nearest_node(self, target_state: RobotState) -> RRTNode:
        """Find the nearest node in the tree to the target state."""
        min_dist = float('inf')
        nearest = self.nodes[0]
        
        for node in self.nodes:
            # Distance metric: combination of anchor position and angle
            pos_dist = np.linalg.norm(node.state.anchor_pos - target_state.anchor_pos)
            angle_dist = abs(self._angle_diff(node.state.angle, target_state.angle))
            dist = pos_dist + 0.5 * angle_dist
            
            if dist < min_dist:
                min_dist = dist
                nearest = node
        
        return nearest
    
    @staticmethod
    def _angle_diff(a1: float, a2: float) -> float:
        """Compute shortest angular difference."""
        diff = a2 - a1
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi
        return diff
    
    def extend(self, from_node: RRTNode, target_state: RobotState) -> Optional[RRTNode]:
        """
        Try to extend the tree from from_node towards target_state.
        
        Returns:
            New node if extension successful, None otherwise
        """
        current = from_node.state
        
        # Try different actions
        candidates = []
        
        # Action 1: Swing towards target angle
        angle_diff = self._angle_diff(current.angle, target_state.angle)
        if abs(angle_diff) > 0.01:
            step = np.sign(angle_diff) * min(abs(angle_diff), self.swing_step)
            new_angle = current.angle + step
            new_state = self.robot.swing_to_angle(current, new_angle)
            
            if self.env.is_robot_state_valid(self.robot, new_state):
                candidates.append((new_state, 'swing'))
        
        # Also try random swing directions to escape local minima
        for _ in range(3):
            random_angle = current.angle + random.uniform(-self.swing_step, self.swing_step)
            new_state = self.robot.swing_to_angle(current, random_angle)
            if self.env.is_robot_state_valid(self.robot, new_state):
                candidates.append((new_state, 'swing'))
        
        # Action 2: Try to switch anchor (if free end is near a hold point)
        free_end = self.robot.get_free_end_position(current)
        
        # Find all nearby hold points
        for hold in self.env.hold_points:
            dist_to_hold = np.linalg.norm(free_end - hold)
            if dist_to_hold < self.hold_snap_distance:
                # Create new state with switched anchor at the hold point
                new_anchor = hold.copy()
                old_anchor = current.anchor_pos
                direction = old_anchor - new_anchor
                new_angle = np.arctan2(direction[1], direction[0])
                new_anchor_end = 'B' if current.anchor_end == 'A' else 'A'
                new_state = RobotState(new_anchor, new_angle, new_anchor_end)
                
                if self.env.is_robot_state_valid(self.robot, new_state):
                    candidates.append((new_state, 'switch'))
        
        # Select best candidate (prioritize switches for exploration, then closest to goal)
        if not candidates:
            return None
        
        # Separate switches and swings
        switches = [(s, a) for s, a in candidates if a == 'switch']
        swings = [(s, a) for s, a in candidates if a == 'swing']
        
        # Prefer switches when they get us closer to goal
        if switches:
            best_switch = min(switches, key=lambda x: self.distance_to_goal(x[0]))
            if self.distance_to_goal(best_switch[0]) < self.distance_to_goal(current):
                return RRTNode(best_switch[0], from_node, best_switch[1])
        
        # Otherwise pick best overall
        best_state, best_action = min(candidates, 
            key=lambda x: self.distance_to_goal(x[0]))
        
        return RRTNode(best_state, from_node, best_action)
    
    def plan(self, start_state: RobotState) -> Optional[List[RRTNode]]:
        """
        Plan a path from start to goal using RRT.
        
        Args:
            start_state: Initial robot state
            
        Returns:
            List of nodes forming the path, or None if no path found
        """
        # Initialize tree
        self.nodes = [RRTNode(start_state)]
        
        print(f"Starting RRT planning...")
        print(f"  Start: anchor={start_state.anchor_pos}, angle={np.degrees(start_state.angle):.1f}°")
        print(f"  Goal: {self.goal_pos}")
        
        best_distance = float('inf')
        
        for iteration in range(self.max_iterations):
            # Sample random state
            random_state = self.sample_random_state()
            
            # Find nearest node
            nearest = self.find_nearest_node(random_state)
            
            # Extend tree
            new_node = self.extend(nearest, random_state)
            
            if new_node is not None:
                self.nodes.append(new_node)
                
                current_dist = self.distance_to_goal(new_node.state)
                if current_dist < best_distance:
                    best_distance = current_dist
                
                # Check if goal reached
                if self.is_goal_reached(new_node.state):
                    print(f"  Goal reached after {iteration + 1} iterations!")
                    return self._extract_path(new_node)
                
                # Greedy expansion: if we just switched, try to keep moving toward goal
                if new_node.action == 'switch':
                    for _ in range(5):  # Try a few more extensions
                        greedy_target = RobotState(
                            self.env.get_nearest_hold(self.goal_pos).copy(),
                            np.arctan2(self.goal_pos[1] - new_node.state.anchor_pos[1],
                                      self.goal_pos[0] - new_node.state.anchor_pos[0]),
                            new_node.state.anchor_end
                        )
                        next_node = self.extend(new_node, greedy_target)
                        if next_node is not None:
                            self.nodes.append(next_node)
                            if self.is_goal_reached(next_node.state):
                                print(f"  Goal reached after {iteration + 1} iterations!")
                                return self._extract_path(next_node)
                            new_node = next_node
            
            # Progress update
            if (iteration + 1) % 500 == 0:
                print(f"  Iteration {iteration + 1}: {len(self.nodes)} nodes, best distance: {best_distance:.2f}")
        
        print(f"  Search ended after {self.max_iterations} iterations")
        # Return best path found
        best_node = min(self.nodes, key=lambda n: self.distance_to_goal(n.state))
        best_dist = self.distance_to_goal(best_node.state)
        if best_dist < self.goal_tolerance * 3:
            print(f"  Returning best partial path (distance: {best_dist:.2f})")
            return self._extract_path(best_node)
        return None
    
    def _extract_path(self, goal_node: RRTNode) -> List[RRTNode]:
        """Extract path from start to goal node."""
        path = []
        current = goal_node
        while current is not None:
            path.append(current)
            current = current.parent
        return list(reversed(path))


# ==============================================================================
# Visualization
# ==============================================================================

class BrachiationVisualizer:
    """Visualization for the brachiation robot simulation."""
    
    def __init__(self, robot: BrachiationRobot, env: Environment):
        self.robot = robot
        self.env = env
        self.fig = None
        self.ax = None
    
    def setup_plot(self, goal_pos: np.ndarray):
        """Setup the matplotlib figure."""
        self.fig, self.ax = plt.subplots(1, 1, figsize=(14, 10))
        self.ax.set_xlim(-1, self.env.width + 1)
        self.ax.set_ylim(-1, self.env.height + 1)
        self.ax.set_aspect('equal')
        self.ax.grid(True, alpha=0.3)
        self.ax.set_xlabel('X Position')
        self.ax.set_ylabel('Y Position')
        self.ax.set_title('2D Brachiation Robot with RRT Path Planning')
        
        # Draw environment
        self._draw_environment(goal_pos)
        
        return self.fig, self.ax
    
    def _draw_environment(self, goal_pos: np.ndarray):
        """Draw obstacles, hold points, and goal."""
        # Draw obstacles
        for obs in self.env.obstacles:
            rect = patches.Rectangle(
                (obs.x, obs.y), obs.width, obs.height,
                linewidth=2, edgecolor='darkgray', facecolor='gray', alpha=0.7
            )
            self.ax.add_patch(rect)
        
        # Draw hold points
        if self.env.hold_points:
            holds = np.array(self.env.hold_points)
            self.ax.scatter(holds[:, 0], holds[:, 1], c='brown', s=50, 
                          marker='o', alpha=0.5, label='Hold Points', zorder=2)
        
        # Draw goal
        goal_circle = plt.Circle(goal_pos, 0.5, color='green', alpha=0.3)
        self.ax.add_patch(goal_circle)
        self.ax.scatter(*goal_pos, c='green', s=200, marker='*', 
                       label='Goal', zorder=5)
    
    def draw_robot(self, state: RobotState, color='blue', alpha=1.0, 
                   linewidth=3, show_endpoints=True):
        """Draw the robot at a given state."""
        anchor, pivot, free_end = self.robot.get_all_positions(state)
        
        # Draw links
        self.ax.plot([anchor[0], pivot[0]], [anchor[1], pivot[1]], 
                    color=color, linewidth=linewidth, alpha=alpha, zorder=3)
        self.ax.plot([pivot[0], free_end[0]], [pivot[1], free_end[1]], 
                    color=color, linewidth=linewidth, alpha=alpha, zorder=3)
        
        if show_endpoints:
            # Draw pivot
            self.ax.scatter(*pivot, c='yellow', s=80, marker='o', 
                          edgecolors='black', zorder=4)
            
            # Draw anchor point (larger, with anchor symbol)
            self.ax.scatter(*anchor, c='red', s=120, marker='s', 
                          edgecolors='black', linewidths=2, zorder=4)
            
            # Draw free end
            self.ax.scatter(*free_end, c='blue', s=100, marker='o', 
                          edgecolors='black', zorder=4)
    
    def draw_rrt_tree(self, nodes: List[RRTNode], alpha=0.2):
        """Draw the RRT tree."""
        for node in nodes:
            if node.parent is not None:
                # Draw connection between states
                p1 = self.robot.get_pivot_position(node.parent.state)
                p2 = self.robot.get_pivot_position(node.state)
                self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                           'gray', linewidth=0.5, alpha=alpha)
    
    def draw_path(self, path: List[RRTNode]):
        """Draw the planned path."""
        if not path:
            return
        
        # Draw path connections
        for i, node in enumerate(path):
            alpha = 0.3 + 0.7 * (i / len(path))  # Fade in
            color = plt.cm.viridis(i / len(path))
            self.draw_robot(node.state, color=color, alpha=alpha, 
                          linewidth=2, show_endpoints=False)
        
        # Highlight start and end
        self.draw_robot(path[0].state, color='green', linewidth=4)
        self.draw_robot(path[-1].state, color='red', linewidth=4)
    
    def animate_path(self, path: List[RRTNode], goal_pos: np.ndarray, 
                     interval: int = 500, save_path: Optional[str] = None):
        """Animate the robot following the path."""
        self.setup_plot(goal_pos)
        
        # Draw ghost path
        for i, node in enumerate(path):
            alpha = 0.15
            self.draw_robot(node.state, color='lightblue', alpha=alpha, 
                          linewidth=1, show_endpoints=False)
        
        # Animation elements
        robot_lines = []
        robot_points = []
        
        def init():
            return []
        
        def animate(frame):
            # Clear previous robot
            for line in robot_lines:
                line.remove()
            for point in robot_points:
                point.remove()
            robot_lines.clear()
            robot_points.clear()
            
            if frame < len(path):
                state = path[frame].state
                anchor, pivot, free_end = self.robot.get_all_positions(state)
                
                # Draw links
                line1, = self.ax.plot([anchor[0], pivot[0]], [anchor[1], pivot[1]], 
                                     color='blue', linewidth=4, zorder=10)
                line2, = self.ax.plot([pivot[0], free_end[0]], [pivot[1], free_end[1]], 
                                     color='blue', linewidth=4, zorder=10)
                robot_lines.extend([line1, line2])
                
                # Draw points
                p1 = self.ax.scatter(*pivot, c='yellow', s=100, marker='o', 
                                    edgecolors='black', zorder=11)
                p2 = self.ax.scatter(*anchor, c='red', s=150, marker='s', 
                                    edgecolors='black', linewidths=2, zorder=11)
                p3 = self.ax.scatter(*free_end, c='cyan', s=120, marker='o', 
                                    edgecolors='black', zorder=11)
                robot_points.extend([p1, p2, p3])
                
                # Update title
                action = path[frame].action if path[frame].action else "Start"
                self.ax.set_title(f'Step {frame + 1}/{len(path)} - Action: {action}')
            
            return robot_lines + robot_points
        
        anim = FuncAnimation(self.fig, animate, init_func=init,
                            frames=len(path), interval=interval, 
                            blit=False, repeat=True)
        
        if save_path:
            anim.save(save_path, writer='pillow', fps=2)
            print(f"Animation saved to {save_path}")
        
        plt.legend(loc='upper right')
        plt.tight_layout()
        return anim


# ==============================================================================
# Main Simulation
# ==============================================================================

def run_simulation(goal_x: float = 17.0, goal_y: float = 12.0):
    """
    Run the brachiation robot simulation.
    
    Args:
        goal_x: X coordinate of goal position
        goal_y: Y coordinate of goal position
    """
    print("=" * 60)
    print("2D Brachiation Robot Simulation with RRT Path Planning")
    print("=" * 60)
    
    # Create robot
    robot = BrachiationRobot(link_length=1.5)
    print(f"\nRobot created with link length: {robot.link_length}")
    print(f"Total span: {robot.total_length}")
    
    # Create environment
    env = Environment(width=20.0, height=15.0)
    env.create_default_environment()
    print(f"\nEnvironment: {env.width}x{env.height}")
    print(f"Obstacles: {len(env.obstacles)}")
    print(f"Hold points: {len(env.hold_points)}")
    
    # Define start and goal
    # Find a valid starting hold point
    start_anchor = np.array([2.5, 2.5])  # Starting anchor position (bottom left area)
    start_angle = np.pi / 4  # Pointing up-right
    start_state = RobotState(start_anchor, start_angle, 'A')
    
    # Adjust start if invalid
    if not env.is_robot_state_valid(robot, start_state):
        print("Initial start invalid, searching for valid start...")
        for hold in env.hold_points:
            for angle in np.linspace(-np.pi, np.pi, 16):
                test_state = RobotState(hold.copy(), angle, 'A')
                if env.is_robot_state_valid(robot, test_state):
                    start_state = test_state
                    start_anchor = hold.copy()
                    start_angle = angle
                    print(f"Found valid start at {hold} with angle {np.degrees(angle):.1f}°")
                    break
            else:
                continue
            break
    
    goal_pos = np.array([goal_x, goal_y])
    
    print(f"\nStart state:")
    print(f"  Anchor: {start_state.anchor_pos}")
    print(f"  Angle: {np.degrees(start_state.angle):.1f}°")
    print(f"  Anchor end: {start_state.anchor_end}")
    
    print(f"\nGoal position: {goal_pos}")
    
    # Verify start state is valid
    if not env.is_robot_state_valid(robot, start_state):
        print("ERROR: Start state is invalid!")
        return
    
    # Create RRT planner
    planner = BrachiationRRT(robot, env, goal_pos, goal_tolerance=1.0)
    
    # Plan path
    print("\n" + "-" * 40)
    start_time = time.time()
    path = planner.plan(start_state)
    planning_time = time.time() - start_time
    print(f"Planning time: {planning_time:.2f} seconds")
    print("-" * 40)
    
    # Visualize
    visualizer = BrachiationVisualizer(robot, env)
    
    if path:
        print(f"\nPath found with {len(path)} steps!")
        print("\nPath summary:")
        for i, node in enumerate(path):
            action = node.action if node.action else "START"
            end_a, end_b = robot.get_end_positions(node.state)
            dist_to_goal = min(np.linalg.norm(end_a - goal_pos),
                              np.linalg.norm(end_b - goal_pos))
            print(f"  Step {i}: {action:6s} | Anchor: ({node.state.anchor_pos[0]:.1f}, {node.state.anchor_pos[1]:.1f}) | "
                  f"Angle: {np.degrees(node.state.angle):6.1f}° | Dist to goal: {dist_to_goal:.2f}")
        
        # Create animation
        print("\nCreating visualization...")
        fig, ax = visualizer.setup_plot(goal_pos)
        
        # Draw RRT tree (optional - can be slow for large trees)
        if len(planner.nodes) < 2000:
            visualizer.draw_rrt_tree(planner.nodes)
        
        # Draw path
        visualizer.draw_path(path)
        
        # Add legend
        ax.legend(loc='upper right')
        
        # Save static image
        plt.savefig('brachiation_rrt_path.png', dpi=150, bbox_inches='tight')
        print("Static path saved to 'brachiation_rrt_path.png'")
        
        # Create animation (only if not headless)
        if not HEADLESS:
            anim = visualizer.animate_path(path, goal_pos, interval=800,
                                           save_path='brachiation_animation.gif')
            plt.show()
        else:
            print("Running in headless mode - skipping animation display")
        
    else:
        print("\nNo path found!")
        # Still show environment
        fig, ax = visualizer.setup_plot(goal_pos)
        visualizer.draw_robot(start_state, color='blue')
        visualizer.draw_rrt_tree(planner.nodes)
        plt.savefig('brachiation_rrt_failed.png', dpi=150, bbox_inches='tight')
        if not HEADLESS:
            plt.show()
    
    print("\n" + "=" * 60)
    print("Simulation complete!")
    print("=" * 60)


def interactive_demo():
    """Interactive demo where user can click to set goal."""
    print("=" * 60)
    print("Interactive Brachiation Demo")
    print("=" * 60)
    print("\nClick on the plot to set the goal position.")
    print("Close the window when done.\n")
    
    # Create robot and environment
    robot = BrachiationRobot(link_length=1.5)
    env = Environment(width=20.0, height=15.0)
    env.create_default_environment()
    
    # Initial visualization
    visualizer = BrachiationVisualizer(robot, env)
    fig, ax = visualizer.setup_plot(np.array([17.0, 12.0]))
    
    # Start state
    start_state = RobotState(np.array([2.0, 7.5]), 0.0, 'A')
    visualizer.draw_robot(start_state, color='blue')
    
    goal_marker = [None]
    
    def onclick(event):
        if event.xdata is None or event.ydata is None:
            return
        
        goal_pos = np.array([event.xdata, event.ydata])
        print(f"\nGoal set to: ({goal_pos[0]:.1f}, {goal_pos[1]:.1f})")
        
        # Clear previous goal marker
        if goal_marker[0] is not None:
            goal_marker[0].remove()
        
        goal_marker[0] = ax.scatter(*goal_pos, c='red', s=300, marker='*', zorder=10)
        fig.canvas.draw()
        
        # Plan and visualize
        planner = BrachiationRRT(robot, env, goal_pos, goal_tolerance=1.0)
        path = planner.plan(start_state)
        
        if path:
            print(f"Path found with {len(path)} steps!")
            ax.clear()
            visualizer.setup_plot(goal_pos)
            visualizer.draw_rrt_tree(planner.nodes)
            visualizer.draw_path(path)
            fig.canvas.draw()
    
    fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='2D Brachiation Robot with RRT')
    parser.add_argument('--goal-x', type=float, default=17.0, 
                        help='Goal X position (default: 17.0)')
    parser.add_argument('--goal-y', type=float, default=12.0, 
                        help='Goal Y position (default: 12.0)')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive mode (click to set goal)')
    parser.add_argument('--headless', action='store_true',
                        help='Run without display (save images only)')
    
    args = parser.parse_args()
    
    if args.interactive:
        interactive_demo()
    else:
        run_simulation(goal_x=args.goal_x, goal_y=args.goal_y)
