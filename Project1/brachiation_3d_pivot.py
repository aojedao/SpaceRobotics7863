#!/usr/bin/env python3
"""
3D Brachiation Robot with Pivot Joint - RRT Path Planning

This robot has TWO links connected by a PIVOT JOINT in the middle.
The robot moves through a 3D walled cubic environment in zero-gravity.

Robot structure:
    End A ----[Link 1]---- Pivot (joint) ----[Link 2]---- End B
    
In 3D, the configuration includes:
- anchor_pos: 3D position of anchored end
- base_theta: azimuth angle of link 1 (rotation in XY plane)
- base_phi: elevation angle of link 1 (angle from XY plane)
- pivot_angle: relative bend at pivot joint

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
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from dataclasses import dataclass
from typing import List, Tuple, Optional
import random
import time


# ==============================================================================
# 3D Robot Configuration
# ==============================================================================

@dataclass
class RobotState3D:
    """
    State of the 3D two-link brachiation robot with pivot joint.
    
    - anchor_pos: 3D position of anchored end (x, y, z)
    - base_theta: azimuth angle of link 1 (XY plane rotation, 0 = +X)
    - base_phi: elevation angle of link 1 (from XY plane, 0 = horizontal)
    - pivot_angle: relative bend at pivot joint (-pi to pi)
    - anchor_end: which end is anchored ('A' or 'B')
    """
    anchor_pos: np.ndarray
    base_theta: float   # Azimuth
    base_phi: float     # Elevation
    pivot_angle: float  # Bend at pivot
    anchor_end: str
    
    def __post_init__(self):
        if isinstance(self.anchor_pos, (list, tuple)):
            self.anchor_pos = np.array(self.anchor_pos, dtype=float)
    
    def copy(self):
        return RobotState3D(
            self.anchor_pos.copy(), 
            self.base_theta, 
            self.base_phi, 
            self.pivot_angle, 
            self.anchor_end
        )


class PivotBrachiationRobot3D:
    """3D two-link brachiation robot with pivot joint."""
    
    def __init__(self, link_length: float = 1.5):
        self.link_length = link_length
        self.pivot_range = (-np.pi * 0.75, np.pi * 0.75)
        self.phi_range = (-np.pi/2 * 0.9, np.pi/2 * 0.9)  # Elevation limits
    
    def _spherical_to_cartesian(self, theta: float, phi: float) -> np.ndarray:
        """Convert spherical angles to unit direction vector."""
        return np.array([
            np.cos(phi) * np.cos(theta),
            np.cos(phi) * np.sin(theta),
            np.sin(phi)
        ])
    
    def get_pivot_position(self, state: RobotState3D) -> np.ndarray:
        """Get position of central pivot."""
        direction = self._spherical_to_cartesian(state.base_theta, state.base_phi)
        return state.anchor_pos + self.link_length * direction
    
    def get_free_end_position(self, state: RobotState3D) -> np.ndarray:
        """Get position of free end."""
        pivot = self.get_pivot_position(state)
        
        # Free end direction: rotate from base direction by pivot_angle
        # We bend in the plane defined by the base direction and "up" (or perpendicular)
        base_dir = self._spherical_to_cartesian(state.base_theta, state.base_phi)
        
        # Create a perpendicular vector for bending
        if abs(state.base_phi) < np.pi/2 - 0.1:
            up = np.array([0, 0, 1])
        else:
            up = np.array([1, 0, 0])
        
        perp = np.cross(base_dir, up)
        if np.linalg.norm(perp) > 1e-6:
            perp = perp / np.linalg.norm(perp)
        else:
            perp = np.array([0, 1, 0])
        
        # Rotate base_dir around perp by pivot_angle
        free_dir = (base_dir * np.cos(state.pivot_angle) + 
                   np.cross(perp, base_dir) * np.sin(state.pivot_angle) +
                   perp * np.dot(perp, base_dir) * (1 - np.cos(state.pivot_angle)))
        
        return pivot + self.link_length * free_dir
    
    def get_all_positions(self, state: RobotState3D) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get anchor, pivot, and free end positions."""
        return (
            state.anchor_pos.copy(),
            self.get_pivot_position(state),
            self.get_free_end_position(state)
        )
    
    def get_end_positions(self, state: RobotState3D) -> Tuple[np.ndarray, np.ndarray]:
        """Get End A and End B positions."""
        anchor, pivot, free_end = self.get_all_positions(state)
        if state.anchor_end == 'A':
            return anchor, free_end
        else:
            return free_end, anchor
    
    def swing_theta(self, state: RobotState3D, delta: float) -> RobotState3D:
        """Rotate in XY plane (azimuth)."""
        return RobotState3D(
            state.anchor_pos.copy(),
            state.base_theta + delta,
            state.base_phi,
            state.pivot_angle,
            state.anchor_end
        )
    
    def swing_phi(self, state: RobotState3D, delta: float) -> RobotState3D:
        """Change elevation angle."""
        new_phi = np.clip(state.base_phi + delta, self.phi_range[0], self.phi_range[1])
        return RobotState3D(
            state.anchor_pos.copy(),
            state.base_theta,
            new_phi,
            state.pivot_angle,
            state.anchor_end
        )
    
    def bend_pivot(self, state: RobotState3D, delta: float) -> RobotState3D:
        """Bend at pivot joint."""
        new_pivot = np.clip(
            state.pivot_angle + delta,
            self.pivot_range[0],
            self.pivot_range[1]
        )
        return RobotState3D(
            state.anchor_pos.copy(),
            state.base_theta,
            state.base_phi,
            new_pivot,
            state.anchor_end
        )
    
    def switch_anchor(self, state: RobotState3D, new_anchor_pos: np.ndarray) -> RobotState3D:
        """Switch anchor to new position."""
        old_pivot = self.get_pivot_position(state)
        old_anchor = state.anchor_pos
        
        # New base direction: from new anchor to old pivot
        new_base_dir = old_pivot - new_anchor_pos
        if np.linalg.norm(new_base_dir) > 1e-6:
            new_base_dir = new_base_dir / np.linalg.norm(new_base_dir)
        else:
            new_base_dir = np.array([1, 0, 0])
        
        # Convert to spherical
        new_theta = np.arctan2(new_base_dir[1], new_base_dir[0])
        new_phi = np.arcsin(np.clip(new_base_dir[2], -1, 1))
        
        # New pivot angle: angle between new_base_dir and direction to old_anchor
        old_dir = old_anchor - old_pivot
        if np.linalg.norm(old_dir) > 1e-6:
            old_dir = old_dir / np.linalg.norm(old_dir)
            dot = np.clip(np.dot(new_base_dir, old_dir), -1, 1)
            new_pivot = np.arccos(dot)
            # Determine sign
            cross = np.cross(new_base_dir, old_dir)
            if cross[2] < 0:
                new_pivot = -new_pivot
        else:
            new_pivot = 0.0
        
        new_pivot = np.clip(new_pivot, self.pivot_range[0], self.pivot_range[1])
        new_anchor_end = 'B' if state.anchor_end == 'A' else 'A'
        
        return RobotState3D(new_anchor_pos.copy(), new_theta, new_phi, new_pivot, new_anchor_end)


# ==============================================================================
# 3D Walled Environment (Cube)
# ==============================================================================

class CubicEnvironment:
    """3D cubic environment with walls."""
    
    def __init__(self, size: float = 12.0):
        self.size = size
        self.hold_points: List[np.ndarray] = []
    
    def create_hold_grid(self, spacing: float = 2.5, margin: float = 1.5):
        """Create 3D grid of hold points."""
        self.hold_points = []
        for x in np.arange(margin, self.size - margin + 0.1, spacing):
            for y in np.arange(margin, self.size - margin + 0.1, spacing):
                for z in np.arange(margin, self.size - margin + 0.1, spacing):
                    self.hold_points.append(np.array([x, y, z]))
    
    def is_point_inside(self, point: np.ndarray) -> bool:
        """Check if point is inside cube."""
        margin = 0.2
        return all(margin < p < self.size - margin for p in point)
    
    def is_robot_valid(self, robot: PivotBrachiationRobot3D, state: RobotState3D) -> bool:
        """Check if robot is valid (all points inside)."""
        anchor, pivot, free_end = robot.get_all_positions(state)
        return all(self.is_point_inside(p) for p in [anchor, pivot, free_end])
    
    def get_nearest_hold(self, point: np.ndarray) -> Optional[np.ndarray]:
        """Get nearest hold point."""
        if not self.hold_points:
            return None
        distances = [np.linalg.norm(h - point) for h in self.hold_points]
        return self.hold_points[np.argmin(distances)]


# ==============================================================================
# 3D RRT Planner
# ==============================================================================

@dataclass
class RRTNode3D:
    """RRT node for 3D planning."""
    state: RobotState3D
    parent: Optional['RRTNode3D'] = None
    action: Optional[str] = None


class PivotRRT3D:
    """RRT planner for 3D pivot robot."""
    
    def __init__(self, robot: PivotBrachiationRobot3D, env: CubicEnvironment,
                 goal_pos: np.ndarray, goal_tolerance: float = 1.5):
        self.robot = robot
        self.env = env
        self.goal_pos = goal_pos
        self.goal_tolerance = goal_tolerance
        
        self.max_iterations = 4000
        self.theta_step = np.pi / 4
        self.phi_step = np.pi / 6
        self.pivot_step = np.pi / 5
        self.hold_snap_dist = 2.5
        self.goal_bias = 0.4
        
        self.nodes: List[RRTNode3D] = []
    
    def distance_to_goal(self, state: RobotState3D) -> float:
        """Distance from either end to goal."""
        end_a, end_b = self.robot.get_end_positions(state)
        return min(np.linalg.norm(end_a - self.goal_pos),
                   np.linalg.norm(end_b - self.goal_pos))
    
    def is_goal_reached(self, state: RobotState3D) -> bool:
        return self.distance_to_goal(state) < self.goal_tolerance
    
    def sample_random_state(self) -> RobotState3D:
        """Sample random state."""
        if random.random() < self.goal_bias and self.env.hold_points:
            nearest = min(self.env.hold_points,
                         key=lambda h: np.linalg.norm(h - self.goal_pos))
            direction = self.goal_pos - nearest
            if np.linalg.norm(direction) > 1e-6:
                direction = direction / np.linalg.norm(direction)
            theta = np.arctan2(direction[1], direction[0]) + random.uniform(-0.5, 0.5)
            phi = np.arcsin(np.clip(direction[2], -1, 1)) + random.uniform(-0.3, 0.3)
            pivot = random.uniform(-np.pi/2, np.pi/2)
            return RobotState3D(nearest.copy(), theta, phi, pivot, random.choice(['A', 'B']))
        
        if self.env.hold_points:
            anchor = random.choice(self.env.hold_points).copy()
        else:
            anchor = np.array([self.env.size/2]*3)
        
        return RobotState3D(
            anchor,
            random.uniform(-np.pi, np.pi),
            random.uniform(-np.pi/3, np.pi/3),
            random.uniform(-np.pi/2, np.pi/2),
            random.choice(['A', 'B'])
        )
    
    def find_nearest_node(self, target: RobotState3D) -> RRTNode3D:
        """Find nearest node."""
        def dist(node):
            pos_d = np.linalg.norm(node.state.anchor_pos - target.anchor_pos)
            theta_d = abs(node.state.base_theta - target.base_theta)
            phi_d = abs(node.state.base_phi - target.base_phi)
            return pos_d + 0.2 * theta_d + 0.2 * phi_d
        return min(self.nodes, key=dist)
    
    def extend(self, from_node: RRTNode3D, target: RobotState3D) -> Optional[RRTNode3D]:
        """Extend tree."""
        current = from_node.state
        candidates = []
        
        # Swing theta
        for _ in range(2):
            delta = random.uniform(-self.theta_step, self.theta_step)
            new_state = self.robot.swing_theta(current, delta)
            if self.env.is_robot_valid(self.robot, new_state):
                candidates.append((new_state, 'swing_theta'))
        
        # Swing phi
        for _ in range(2):
            delta = random.uniform(-self.phi_step, self.phi_step)
            new_state = self.robot.swing_phi(current, delta)
            if self.env.is_robot_valid(self.robot, new_state):
                candidates.append((new_state, 'swing_phi'))
        
        # Bend pivot
        for _ in range(2):
            delta = random.uniform(-self.pivot_step, self.pivot_step)
            new_state = self.robot.bend_pivot(current, delta)
            if self.env.is_robot_valid(self.robot, new_state):
                candidates.append((new_state, 'bend'))
        
        # Switch anchor
        free_end = self.robot.get_free_end_position(current)
        for hold in self.env.hold_points:
            if np.linalg.norm(free_end - hold) < self.hold_snap_dist:
                new_state = self.robot.switch_anchor(current, hold)
                if self.env.is_robot_valid(self.robot, new_state):
                    candidates.append((new_state, 'switch'))
        
        if not candidates:
            return None
        
        # Prefer switches that get closer
        switches = [(s, a) for s, a in candidates if a == 'switch']
        if switches:
            best = min(switches, key=lambda x: self.distance_to_goal(x[0]))
            if self.distance_to_goal(best[0]) < self.distance_to_goal(current) - 0.2:
                return RRTNode3D(best[0], from_node, best[1])
        
        best = min(candidates, key=lambda x: self.distance_to_goal(x[0]))
        return RRTNode3D(best[0], from_node, best[1])
    
    def plan(self, start_state: RobotState3D) -> Optional[List[RRTNode3D]]:
        """Plan path."""
        self.nodes = [RRTNode3D(start_state)]
        best_dist = float('inf')
        
        print(f"  Planning 3D: start={start_state.anchor_pos}, goal={self.goal_pos}")
        
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
                
                # Greedy after switch
                if new_node.action == 'switch':
                    for _ in range(5):
                        greedy = self.extend(new_node, self.sample_random_state())
                        if greedy:
                            self.nodes.append(greedy)
                            if self.is_goal_reached(greedy.state):
                                print(f"  Goal reached in {iteration+1} iterations!")
                                return self._extract_path(greedy)
                            new_node = greedy
            
            if (iteration + 1) % 1000 == 0:
                print(f"    Iter {iteration+1}: {len(self.nodes)} nodes, best={best_dist:.2f}")
        
        best_node = min(self.nodes, key=lambda n: self.distance_to_goal(n.state))
        print(f"  Partial path, dist={self.distance_to_goal(best_node.state):.2f}")
        return self._extract_path(best_node)
    
    def _extract_path(self, goal_node: RRTNode3D) -> List[RRTNode3D]:
        path = []
        current = goal_node
        while current:
            path.append(current)
            current = current.parent
        return list(reversed(path))


# ==============================================================================
# 3D Visualization
# ==============================================================================

def draw_cube_wireframe(ax, size):
    """Draw cube wireframe."""
    corners = [
        [0, 0, 0], [size, 0, 0], [size, size, 0], [0, size, 0],
        [0, 0, size], [size, 0, size], [size, size, size], [0, size, size]
    ]
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top
        (0, 4), (1, 5), (2, 6), (3, 7)   # Verticals
    ]
    for i, j in edges:
        ax.plot3D(*zip(corners[i], corners[j]), 'k-', linewidth=1, alpha=0.5)


def draw_robot_3d(ax, robot, state, color='blue', alpha=1.0, lw=3):
    """Draw robot in 3D."""
    anchor, pivot, free_end = robot.get_all_positions(state)
    
    # Links
    ax.plot3D([anchor[0], pivot[0]], [anchor[1], pivot[1]], [anchor[2], pivot[2]],
             color=color, linewidth=lw, alpha=alpha)
    ax.plot3D([pivot[0], free_end[0]], [pivot[1], free_end[1]], [pivot[2], free_end[2]],
             color=color, linewidth=lw, alpha=alpha)
    
    # Joints
    ax.scatter(*pivot, c='yellow', s=60, marker='o', edgecolors='black')
    ax.scatter(*anchor, c='red', s=80, marker='s', edgecolors='black')
    ax.scatter(*free_end, c='cyan', s=60, marker='o', edgecolors='black')


def test_3d_goal(robot, env, start_state, goal_pos, goal_num):
    """Test a 3D goal."""
    print(f"\n{'='*50}")
    print(f"3D GOAL {goal_num}: {goal_pos}")
    print('='*50)
    
    planner = PivotRRT3D(robot, env, goal_pos, goal_tolerance=1.5)
    
    t0 = time.time()
    path = planner.plan(start_state)
    dt = time.time() - t0
    
    if path:
        final_dist = planner.distance_to_goal(path[-1].state)
        success = final_dist < planner.goal_tolerance
        status = "SUCCESS" if success else f"PARTIAL (dist={final_dist:.2f})"
        print(f"  Result: {status}")
        print(f"  Path: {len(path)} steps, time: {dt:.2f}s")
        return path, planner, path[-1].state.copy(), success
    return None, planner, start_state, False


def run_3d_multi_goal_test():
    """Run 3D multi-goal test."""
    print("\n" + "="*60)
    print("3D PIVOT BRACHIATION ROBOT - MULTI-GOAL TEST")
    print("="*60)
    
    robot = PivotBrachiationRobot3D(link_length=1.5)
    env = CubicEnvironment(size=12.0)
    env.create_hold_grid(spacing=2.5, margin=2.0)
    
    print(f"\nEnvironment: {env.size}x{env.size}x{env.size} cube")
    print(f"Hold points: {len(env.hold_points)}")
    
    # Start state
    start_hold = env.hold_points[0]
    start_state = RobotState3D(start_hold.copy(), 0.0, 0.0, 0.0, 'A')
    
    if not env.is_robot_valid(robot, start_state):
        print("Finding valid start...")
        for hold in env.hold_points:
            test = RobotState3D(hold.copy(), 0.0, 0.0, 0.0, 'A')
            if env.is_robot_valid(robot, test):
                start_state = test
                break
    
    # 3 goals in 3D
    goals = [
        np.array([10.0, 10.0, 10.0]),  # Top corner
        np.array([2.0, 10.0, 6.0]),    # Side
        np.array([10.0, 2.0, 2.0]),    # Bottom corner
    ]
    
    all_paths = []
    current_state = start_state
    successes = 0
    
    for i, goal in enumerate(goals):
        path, planner, current_state, success = test_3d_goal(
            robot, env, current_state, goal, i+1
        )
        if path:
            all_paths.append((path, goal))
            if success:
                successes += 1
    
    print(f"\n{'='*60}")
    print(f"3D SUMMARY: {successes}/{len(goals)} goals reached")
    print('='*60)
    
    # Visualization
    fig = plt.figure(figsize=(18, 6))
    
    for i, (path, goal) in enumerate(all_paths):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        ax.set_title(f'3D Goal {i+1}: ({goal[0]:.0f}, {goal[1]:.0f}, {goal[2]:.0f})')
        
        draw_cube_wireframe(ax, env.size)
        
        # Goal
        ax.scatter(*goal, c='green', s=200, marker='*')
        
        # Path
        for j, node in enumerate(path):
            alpha = 0.2 + 0.8 * (j / max(len(path)-1, 1))
            color = plt.cm.viridis(j / max(len(path)-1, 1))
            draw_robot_3d(ax, robot, node.state, color=color, alpha=alpha, lw=2)
        
        # Start/end
        draw_robot_3d(ax, robot, path[0].state, color='green', lw=4)
        draw_robot_3d(ax, robot, path[-1].state, color='red', lw=4)
        
        ax.set_xlim(0, env.size)
        ax.set_ylim(0, env.size)
        ax.set_zlim(0, env.size)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        
        ax.text2D(0.02, 0.98, f'Steps: {len(path)}', transform=ax.transAxes)
    
    plt.tight_layout()
    plt.savefig('brachiation_3d_multigoal.png', dpi=150, bbox_inches='tight')
    print("\nSaved: brachiation_3d_multigoal.png")
    
    if not HEADLESS:
        plt.show()
    
    return successes, len(goals)


# ==============================================================================
# Main
# ==============================================================================

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='3D Pivot Brachiation Robot')
    parser.add_argument('--headless', action='store_true')
    args = parser.parse_args()
    
    run_3d_multi_goal_test()
