#!/usr/bin/env python3
"""
2D Brachiation Robot with Pivot Joint - RRT Path Planning

This robot has TWO links connected by a PIVOT JOINT in the middle.
The pivot allows the two links to rotate relative to each other.
The robot moves through a walled square environment in zero-gravity.

Robot structure:
    End A ----[Link 1]---- Pivot (joint) ----[Link 2]---- End B
    
The pivot joint angle determines the relative angle between the two links.
When anchored at one end, both the overall orientation AND the pivot angle
can change.

Author: Space Robotics Project
Date: 2024
"""

import os
import sys

HEADLESS = '--headless' in sys.argv or os.environ.get('DISPLAY') is None

import matplotlib
if HEADLESS:
    matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
from dataclasses import dataclass
from typing import List, Tuple, Optional
import random
import time


# ==============================================================================
# Robot Configuration with Pivot Joint
# ==============================================================================

@dataclass
class RobotState:
    """
    State of the two-link brachiation robot with pivot joint.
    
    - anchor_pos: Position of the currently anchored end (x, y)
    - base_angle: Angle of link 1 from anchor (radians, 0 = right, CCW positive)
    - pivot_angle: Relative angle of link 2 from link 1 at pivot (radians)
    - anchor_end: Which end is anchored ('A' or 'B')
    
    Total robot configuration is determined by:
    - Anchor position (where it's holding)
    - Base angle (orientation of first link)
    - Pivot angle (bend at the middle joint)
    """
    anchor_pos: np.ndarray
    base_angle: float      # Angle of link 1 from anchor
    pivot_angle: float     # Relative angle at pivot joint (-pi to pi)
    anchor_end: str        # 'A' or 'B'
    
    def __post_init__(self):
        if isinstance(self.anchor_pos, (list, tuple)):
            self.anchor_pos = np.array(self.anchor_pos, dtype=float)
    
    def copy(self):
        return RobotState(self.anchor_pos.copy(), self.base_angle, self.pivot_angle, self.anchor_end)
    
    def __hash__(self):
        return hash((tuple(self.anchor_pos), self.base_angle, self.pivot_angle, self.anchor_end))


class PivotBrachiationRobot:
    """
    Two-link brachiation robot with a pivot joint in the middle.
    
    Structure:
        End A ----[Link 1]---- Pivot ----[Link 2]---- End B
        
    The pivot joint allows link 2 to rotate relative to link 1.
    """
    
    def __init__(self, link_length: float = 1.5):
        self.link_length = link_length
        self.pivot_range = (-np.pi * 0.8, np.pi * 0.8)  # Pivot joint limits
    
    def get_pivot_position(self, state: RobotState) -> np.ndarray:
        """Get position of the central pivot joint."""
        offset = self.link_length * np.array([
            np.cos(state.base_angle), 
            np.sin(state.base_angle)
        ])
        return state.anchor_pos + offset
    
    def get_free_end_position(self, state: RobotState) -> np.ndarray:
        """Get position of the free (non-anchored) end."""
        pivot = self.get_pivot_position(state)
        # Free end angle is base_angle + pivot_angle
        free_end_angle = state.base_angle + state.pivot_angle
        offset = self.link_length * np.array([
            np.cos(free_end_angle), 
            np.sin(free_end_angle)
        ])
        return pivot + offset
    
    def get_all_positions(self, state: RobotState) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get positions of anchor, pivot, and free end."""
        anchor = state.anchor_pos.copy()
        pivot = self.get_pivot_position(state)
        free_end = self.get_free_end_position(state)
        return anchor, pivot, free_end
    
    def get_end_positions(self, state: RobotState) -> Tuple[np.ndarray, np.ndarray]:
        """Get positions of End A and End B."""
        anchor, pivot, free_end = self.get_all_positions(state)
        if state.anchor_end == 'A':
            return anchor, free_end
        else:
            return free_end, anchor
    
    def swing_base(self, state: RobotState, delta_angle: float) -> RobotState:
        """Swing the robot by changing base angle (pivot stays fixed)."""
        new_base = state.base_angle + delta_angle
        return RobotState(state.anchor_pos.copy(), new_base, state.pivot_angle, state.anchor_end)
    
    def bend_pivot(self, state: RobotState, delta_angle: float) -> RobotState:
        """Bend the pivot joint by changing pivot angle."""
        new_pivot = np.clip(
            state.pivot_angle + delta_angle, 
            self.pivot_range[0], 
            self.pivot_range[1]
        )
        return RobotState(state.anchor_pos.copy(), state.base_angle, new_pivot, state.anchor_end)
    
    def switch_anchor(self, state: RobotState, new_anchor_pos: np.ndarray) -> RobotState:
        """
        Switch which end is anchored.
        The free end becomes anchored at new_anchor_pos.
        """
        old_anchor = state.anchor_pos
        old_pivot = self.get_pivot_position(state)
        
        # New base angle: from new anchor to pivot
        direction_to_pivot = old_pivot - new_anchor_pos
        new_base_angle = np.arctan2(direction_to_pivot[1], direction_to_pivot[0])
        
        # New pivot angle: from new link1 to old anchor direction
        direction_to_old_anchor = old_anchor - old_pivot
        old_anchor_angle = np.arctan2(direction_to_old_anchor[1], direction_to_old_anchor[0])
        new_pivot_angle = self._normalize_angle(old_anchor_angle - new_base_angle)
        
        # Clamp pivot angle
        new_pivot_angle = np.clip(new_pivot_angle, self.pivot_range[0], self.pivot_range[1])
        
        new_anchor_end = 'B' if state.anchor_end == 'A' else 'A'
        return RobotState(new_anchor_pos.copy(), new_base_angle, new_pivot_angle, new_anchor_end)
    
    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """Normalize angle to [-pi, pi]."""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle


# ==============================================================================
# Walled Square Environment
# ==============================================================================

class WalledEnvironment:
    """Square environment with walls - robot moves INSIDE the walls."""
    
    def __init__(self, size: float = 15.0, wall_thickness: float = 0.3):
        self.size = size
        self.wall_thickness = wall_thickness
        self.hold_points: List[np.ndarray] = []
        
        # Create walls as line segments (for collision checking)
        self.walls = [
            (np.array([0, 0]), np.array([size, 0])),           # Bottom
            (np.array([size, 0]), np.array([size, size])),     # Right
            (np.array([size, size]), np.array([0, size])),     # Top
            (np.array([0, size]), np.array([0, 0])),           # Left
        ]
    
    def add_hold_point(self, x: float, y: float):
        """Add a grasping point."""
        self.hold_points.append(np.array([x, y]))
    
    def create_hold_grid(self, spacing: float = 2.0, margin: float = 1.0):
        """Create a grid of hold points inside the walls."""
        self.hold_points = []
        for x in np.arange(margin, self.size - margin + 0.1, spacing):
            for y in np.arange(margin, self.size - margin + 0.1, spacing):
                self.add_hold_point(x, y)
    
    def is_point_inside(self, point: np.ndarray) -> bool:
        """Check if point is inside the walled area."""
        margin = 0.1
        return (margin < point[0] < self.size - margin and 
                margin < point[1] < self.size - margin)
    
    def segment_intersects_wall(self, p1: np.ndarray, p2: np.ndarray) -> bool:
        """Check if line segment intersects any wall."""
        for w1, w2 in self.walls:
            if self._segments_intersect(p1, p2, w1, w2):
                return True
        return False
    
    @staticmethod
    def _segments_intersect(p1, p2, p3, p4) -> bool:
        """Check if two line segments intersect."""
        def ccw(A, B, C):
            return (C[1] - A[1]) * (B[0] - A[0]) > (B[1] - A[1]) * (C[0] - A[0])
        
        return (ccw(p1, p3, p4) != ccw(p2, p3, p4) and 
                ccw(p1, p2, p3) != ccw(p1, p2, p4))
    
    def is_robot_valid(self, robot: PivotBrachiationRobot, state: RobotState) -> bool:
        """Check if robot configuration is valid (inside walls, no collision)."""
        anchor, pivot, free_end = robot.get_all_positions(state)
        
        # All points must be inside
        if not all(self.is_point_inside(p) for p in [anchor, pivot, free_end]):
            return False
        
        # Links must not intersect walls
        if self.segment_intersects_wall(anchor, pivot):
            return False
        if self.segment_intersects_wall(pivot, free_end):
            return False
        
        return True
    
    def get_nearest_hold(self, point: np.ndarray) -> Optional[np.ndarray]:
        """Get nearest hold point."""
        if not self.hold_points:
            return None
        distances = [np.linalg.norm(h - point) for h in self.hold_points]
        return self.hold_points[np.argmin(distances)]


# ==============================================================================
# RRT Planner for Pivot Robot
# ==============================================================================

@dataclass 
class RRTNode:
    """Node in the RRT tree."""
    state: RobotState
    parent: Optional['RRTNode'] = None
    action: Optional[str] = None


class PivotRRT:
    """RRT planner for the pivot brachiation robot."""
    
    def __init__(self, robot: PivotBrachiationRobot, env: WalledEnvironment,
                 goal_pos: np.ndarray, goal_tolerance: float = 1.0):
        self.robot = robot
        self.env = env
        self.goal_pos = goal_pos
        self.goal_tolerance = goal_tolerance
        
        # RRT parameters
        self.max_iterations = 3000
        self.base_step = np.pi / 4
        self.pivot_step = np.pi / 6
        self.hold_snap_distance = 2.0
        self.goal_bias = 0.35
        
        self.nodes: List[RRTNode] = []
    
    def distance_to_goal(self, state: RobotState) -> float:
        """Distance from either end to goal."""
        end_a, end_b = self.robot.get_end_positions(state)
        return min(np.linalg.norm(end_a - self.goal_pos),
                   np.linalg.norm(end_b - self.goal_pos))
    
    def is_goal_reached(self, state: RobotState) -> bool:
        """Check if goal is reached."""
        return self.distance_to_goal(state) < self.goal_tolerance
    
    def sample_random_state(self) -> RobotState:
        """Sample a random state."""
        if random.random() < self.goal_bias and self.env.hold_points:
            # Bias toward goal
            nearest = min(self.env.hold_points, 
                         key=lambda h: np.linalg.norm(h - self.goal_pos))
            direction = self.goal_pos - nearest
            base_angle = np.arctan2(direction[1], direction[0])
            base_angle += random.uniform(-np.pi/3, np.pi/3)
            pivot_angle = random.uniform(-np.pi/2, np.pi/2)
            return RobotState(nearest.copy(), base_angle, pivot_angle, random.choice(['A', 'B']))
        
        # Random state
        if self.env.hold_points:
            anchor = random.choice(self.env.hold_points).copy()
        else:
            anchor = np.array([self.env.size/2, self.env.size/2])
        
        return RobotState(
            anchor,
            random.uniform(-np.pi, np.pi),
            random.uniform(self.robot.pivot_range[0], self.robot.pivot_range[1]),
            random.choice(['A', 'B'])
        )
    
    def find_nearest_node(self, target: RobotState) -> RRTNode:
        """Find nearest node in tree."""
        def dist(node):
            pos_d = np.linalg.norm(node.state.anchor_pos - target.anchor_pos)
            base_d = abs(self._angle_diff(node.state.base_angle, target.base_angle))
            pivot_d = abs(node.state.pivot_angle - target.pivot_angle)
            return pos_d + 0.3 * base_d + 0.2 * pivot_d
        
        return min(self.nodes, key=dist)
    
    @staticmethod
    def _angle_diff(a1: float, a2: float) -> float:
        """Shortest angular difference."""
        diff = a2 - a1
        while diff > np.pi: diff -= 2*np.pi
        while diff < -np.pi: diff += 2*np.pi
        return diff
    
    def extend(self, from_node: RRTNode, target: RobotState) -> Optional[RRTNode]:
        """Try to extend tree toward target."""
        current = from_node.state
        candidates = []
        
        # Action 1: Swing base
        base_diff = self._angle_diff(current.base_angle, target.base_angle)
        if abs(base_diff) > 0.01:
            step = np.sign(base_diff) * min(abs(base_diff), self.base_step)
            new_state = self.robot.swing_base(current, step)
            if self.env.is_robot_valid(self.robot, new_state):
                candidates.append((new_state, 'swing'))
        
        # Random swing
        for _ in range(2):
            step = random.uniform(-self.base_step, self.base_step)
            new_state = self.robot.swing_base(current, step)
            if self.env.is_robot_valid(self.robot, new_state):
                candidates.append((new_state, 'swing'))
        
        # Action 2: Bend pivot
        pivot_diff = target.pivot_angle - current.pivot_angle
        if abs(pivot_diff) > 0.01:
            step = np.sign(pivot_diff) * min(abs(pivot_diff), self.pivot_step)
            new_state = self.robot.bend_pivot(current, step)
            if self.env.is_robot_valid(self.robot, new_state):
                candidates.append((new_state, 'bend'))
        
        # Random bend
        for _ in range(2):
            step = random.uniform(-self.pivot_step, self.pivot_step)
            new_state = self.robot.bend_pivot(current, step)
            if self.env.is_robot_valid(self.robot, new_state):
                candidates.append((new_state, 'bend'))
        
        # Action 3: Switch anchor
        free_end = self.robot.get_free_end_position(current)
        for hold in self.env.hold_points:
            if np.linalg.norm(free_end - hold) < self.hold_snap_distance:
                new_state = self.robot.switch_anchor(current, hold)
                if self.env.is_robot_valid(self.robot, new_state):
                    candidates.append((new_state, 'switch'))
        
        if not candidates:
            return None
        
        # Prefer switches that reduce distance to goal
        switches = [(s, a) for s, a in candidates if a == 'switch']
        if switches:
            best_switch = min(switches, key=lambda x: self.distance_to_goal(x[0]))
            if self.distance_to_goal(best_switch[0]) < self.distance_to_goal(current) - 0.1:
                return RRTNode(best_switch[0], from_node, best_switch[1])
        
        # Otherwise pick best
        best = min(candidates, key=lambda x: self.distance_to_goal(x[0]))
        return RRTNode(best[0], from_node, best[1])
    
    def plan(self, start_state: RobotState) -> Optional[List[RRTNode]]:
        """Plan path from start to goal."""
        self.nodes = [RRTNode(start_state)]
        best_dist = float('inf')
        
        print(f"Planning: start={start_state.anchor_pos}, goal={self.goal_pos}")
        
        for iteration in range(self.max_iterations):
            random_state = self.sample_random_state()
            nearest = self.find_nearest_node(random_state)
            new_node = self.extend(nearest, random_state)
            
            if new_node:
                self.nodes.append(new_node)
                dist = self.distance_to_goal(new_node.state)
                if dist < best_dist:
                    best_dist = dist
                
                if self.is_goal_reached(new_node.state):
                    print(f"  Goal reached in {iteration+1} iterations!")
                    return self._extract_path(new_node)
                
                # Greedy extension after switch
                if new_node.action == 'switch':
                    for _ in range(5):
                        greedy = self.extend(new_node, self.sample_random_state())
                        if greedy:
                            self.nodes.append(greedy)
                            if self.is_goal_reached(greedy.state):
                                print(f"  Goal reached in {iteration+1} iterations!")
                                return self._extract_path(greedy)
                            new_node = greedy
            
            if (iteration + 1) % 500 == 0:
                print(f"  Iter {iteration+1}: {len(self.nodes)} nodes, best_dist={best_dist:.2f}")
        
        # Return best partial path
        best_node = min(self.nodes, key=lambda n: self.distance_to_goal(n.state))
        print(f"  Returning partial path, dist={self.distance_to_goal(best_node.state):.2f}")
        return self._extract_path(best_node)
    
    def _extract_path(self, goal_node: RRTNode) -> List[RRTNode]:
        """Extract path from start to goal."""
        path = []
        current = goal_node
        while current:
            path.append(current)
            current = current.parent
        return list(reversed(path))


# ==============================================================================
# Visualization
# ==============================================================================

class Visualizer:
    """Visualization for the pivot robot."""
    
    def __init__(self, robot: PivotBrachiationRobot, env: WalledEnvironment):
        self.robot = robot
        self.env = env
    
    def setup_plot(self, goal_pos: np.ndarray, title: str = ""):
        """Setup the plot."""
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.set_xlim(-0.5, self.env.size + 0.5)
        ax.set_ylim(-0.5, self.env.size + 0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_title(f'2D Pivot Brachiation Robot - RRT {title}')
        
        # Draw walls
        for w1, w2 in self.env.walls:
            ax.plot([w1[0], w2[0]], [w1[1], w2[1]], 'k-', linewidth=4)
        
        # Draw hold points
        if self.env.hold_points:
            holds = np.array(self.env.hold_points)
            ax.scatter(holds[:, 0], holds[:, 1], c='brown', s=40, 
                      marker='o', alpha=0.4, label='Holds')
        
        # Draw goal
        goal_circle = plt.Circle(goal_pos, self.env.size * 0.05, color='green', alpha=0.3)
        ax.add_patch(goal_circle)
        ax.scatter(*goal_pos, c='green', s=200, marker='*', label='Goal', zorder=10)
        
        return fig, ax
    
    def draw_robot(self, ax, state: RobotState, color='blue', alpha=1.0, lw=3):
        """Draw robot at state."""
        anchor, pivot, free_end = self.robot.get_all_positions(state)
        
        # Draw links
        ax.plot([anchor[0], pivot[0]], [anchor[1], pivot[1]], 
               color=color, linewidth=lw, alpha=alpha)
        ax.plot([pivot[0], free_end[0]], [pivot[1], free_end[1]], 
               color=color, linewidth=lw, alpha=alpha)
        
        # Draw joints
        ax.scatter(*pivot, c='yellow', s=80, marker='o', edgecolors='black', zorder=5)
        ax.scatter(*anchor, c='red', s=100, marker='s', edgecolors='black', zorder=5)
        ax.scatter(*free_end, c='cyan', s=80, marker='o', edgecolors='black', zorder=5)
    
    def draw_path(self, ax, path: List[RRTNode]):
        """Draw the planned path."""
        for i, node in enumerate(path):
            alpha = 0.2 + 0.8 * (i / max(len(path)-1, 1))
            color = plt.cm.viridis(i / max(len(path)-1, 1))
            self.draw_robot(ax, node.state, color=color, alpha=alpha, lw=2)
        
        # Highlight start and end
        if path:
            self.draw_robot(ax, path[0].state, color='green', lw=4)
            self.draw_robot(ax, path[-1].state, color='red', lw=4)


# ==============================================================================
# Main
# ==============================================================================

def test_goal(robot, env, start_state, goal_pos, goal_num):
    """Test reaching a specific goal."""
    print(f"\n{'='*50}")
    print(f"GOAL {goal_num}: {goal_pos}")
    print('='*50)
    
    planner = PivotRRT(robot, env, goal_pos, goal_tolerance=1.0)
    
    t0 = time.time()
    path = planner.plan(start_state)
    dt = time.time() - t0
    
    if path:
        final_dist = planner.distance_to_goal(path[-1].state)
        success = final_dist < planner.goal_tolerance
        status = "SUCCESS" if success else f"PARTIAL (dist={final_dist:.2f})"
        print(f"Result: {status}")
        print(f"Path length: {len(path)} steps")
        print(f"Planning time: {dt:.2f}s")
        
        # Get next start state (from end of this path)
        next_start = path[-1].state.copy()
        
        return path, planner, next_start, success
    else:
        print("No path found!")
        return None, planner, start_state, False


def run_multi_goal_test():
    """Test robot reaching multiple goals."""
    print("="*60)
    print("2D PIVOT BRACHIATION ROBOT - MULTI-GOAL TEST")
    print("="*60)
    
    # Create robot and environment
    robot = PivotBrachiationRobot(link_length=1.5)
    env = WalledEnvironment(size=15.0)
    env.create_hold_grid(spacing=2.0, margin=1.5)
    
    print(f"\nEnvironment: {env.size}x{env.size} walled square")
    print(f"Hold points: {len(env.hold_points)}")
    print(f"Robot link length: {robot.link_length}")
    
    # Initial state
    start_hold = env.hold_points[0]
    start_state = RobotState(start_hold.copy(), np.pi/4, 0.0, 'A')
    
    # Ensure valid start
    if not env.is_robot_valid(robot, start_state):
        print("Finding valid start...")
        for hold in env.hold_points:
            for base_ang in np.linspace(0, 2*np.pi, 8):
                test = RobotState(hold.copy(), base_ang, 0.0, 'A')
                if env.is_robot_valid(robot, test):
                    start_state = test
                    break
            else:
                continue
            break
    
    print(f"Start: anchor={start_state.anchor_pos}")
    
    # Define 3 goals
    goals = [
        np.array([12.0, 12.0]),  # Top-right
        np.array([3.0, 12.0]),   # Top-left  
        np.array([12.0, 3.0]),   # Bottom-right
    ]
    
    all_paths = []
    all_planners = []
    current_state = start_state
    successes = 0
    
    for i, goal in enumerate(goals):
        path, planner, current_state, success = test_goal(
            robot, env, current_state, goal, i+1
        )
        if path:
            all_paths.append(path)
            all_planners.append(planner)
            if success:
                successes += 1
    
    print(f"\n{'='*60}")
    print(f"SUMMARY: {successes}/{len(goals)} goals reached")
    print('='*60)
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    for i, (path, goal) in enumerate(zip(all_paths, goals)):
        ax = axes[i]
        vis = Visualizer(robot, env)
        
        # Setup
        ax.set_xlim(-0.5, env.size + 0.5)
        ax.set_ylim(-0.5, env.size + 0.5)
        ax.set_aspect('equal')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'Goal {i+1}: ({goal[0]:.0f}, {goal[1]:.0f})')
        
        # Walls
        for w1, w2 in env.walls:
            ax.plot([w1[0], w2[0]], [w1[1], w2[1]], 'k-', linewidth=3)
        
        # Holds
        holds = np.array(env.hold_points)
        ax.scatter(holds[:, 0], holds[:, 1], c='brown', s=20, alpha=0.3)
        
        # Goal
        ax.scatter(*goal, c='green', s=200, marker='*', zorder=10)
        goal_circle = plt.Circle(goal, 1.0, color='green', alpha=0.2)
        ax.add_patch(goal_circle)
        
        # Path
        vis.draw_path(ax, path)
        
        # Info
        ax.text(0.02, 0.98, f'Steps: {len(path)}', transform=ax.transAxes,
               fontsize=10, verticalalignment='top')
    
    plt.tight_layout()
    plt.savefig('brachiation_pivot_multigoal.png', dpi=150, bbox_inches='tight')
    print("\nSaved: brachiation_pivot_multigoal.png")
    
    if not HEADLESS:
        plt.show()
    
    return successes, len(goals)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='2D Pivot Brachiation Robot')
    parser.add_argument('--headless', action='store_true', help='No display')
    args = parser.parse_args()
    
    run_multi_goal_test()
