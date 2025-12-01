#!/usr/bin/env python3
"""
Dual-Arm Wall-Crawler A* Path Planner with 2D Animation

This script simulates a bipedal wall-crawling robot (like a snail or crawler)
that moves ONLY along the walls (perimeter) of a container box.

The robot has two "feet" (arms) and moves with static stability:
- One foot stays fixed on the wall
- The other foot moves to the next position
- Then switch roles

Movement is constrained to the walls of the container only.
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive display
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import heapq
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set, Dict
import time


@dataclass
class CrawlerState:
    """State representation for the wall-crawler robot"""
    position: Tuple[float, float]  # Current position on the wall
    wall: str  # Which wall: 'bottom', 'right', 'top', 'left'
    holding_foot: int  # Which foot is currently holding (0=back, 1=front)
    
    def __hash__(self):
        return hash((self.position, self.wall, self.holding_foot))
    
    def __eq__(self, other):
        return (self.position == other.position and 
                self.wall == other.wall and 
                self.holding_foot == other.holding_foot)
    
    def __lt__(self, other):
        return self.position < other.position


@dataclass(order=True)
class PriorityNode:
    """Node for A* priority queue"""
    f_cost: float
    state: CrawlerState = field(compare=False)
    g_cost: float = field(compare=False)
    parent: Optional['PriorityNode'] = field(compare=False, default=None)


class WallCrawlerPlanner:
    """
    Wall-crawling bipedal robot planner.
    
    The robot moves only along the walls (perimeter) of a rectangular container.
    It uses two feet for static stability - one holds while the other moves.
    """
    
    def __init__(self, 
                 container_size: Tuple[float, float] = (2.0, 2.0),
                 step_size: float = 0.15,
                 foot_reach: float = 0.25):
        """
        Initialize the wall crawler planner.
        
        Args:
            container_size: (width, height) of the container in meters
            step_size: Distance between discrete positions on the wall
            foot_reach: Maximum reach of each foot
        """
        self.width, self.height = container_size
        self.step_size = step_size
        self.foot_reach = foot_reach
        
        # Create wall positions (discrete points along perimeter)
        self.wall_positions = self._create_wall_positions()
        
        # Wall neighbors for navigation
        self.position_to_index = {pos: i for i, (pos, wall) in enumerate(self.wall_positions)}
        
        print(f"Created {len(self.wall_positions)} positions along the perimeter")
        
    def _create_wall_positions(self) -> List[Tuple[Tuple[float, float], str]]:
        """Create discrete positions along all four walls"""
        positions = []
        
        # Bottom wall (y = 0, x goes from 0 to width)
        x = 0.0
        while x <= self.width:
            positions.append(((x, 0.0), 'bottom'))
            x += self.step_size
        
        # Right wall (x = width, y goes from 0 to height) - skip corner
        y = self.step_size
        while y <= self.height:
            positions.append(((self.width, y), 'right'))
            y += self.step_size
        
        # Top wall (y = height, x goes from width to 0) - skip corner
        x = self.width - self.step_size
        while x >= 0:
            positions.append(((x, self.height), 'top'))
            x -= self.step_size
        
        # Left wall (x = 0, y goes from height to 0) - skip corners
        y = self.height - self.step_size
        while y > 0:
            positions.append(((0.0, y), 'left'))
            y -= self.step_size
        
        return positions
    
    def get_wall_for_position(self, pos: Tuple[float, float]) -> str:
        """Determine which wall a position is on"""
        x, y = pos
        eps = 0.01
        
        if abs(y) < eps:
            return 'bottom'
        elif abs(x - self.width) < eps:
            return 'right'
        elif abs(y - self.height) < eps:
            return 'top'
        elif abs(x) < eps:
            return 'left'
        return 'unknown'
    
    def get_distance(self, pos1: Tuple[float, float], pos2: Tuple[float, float]) -> float:
        """Euclidean distance between two positions"""
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
    
    def get_wall_distance(self, pos1: Tuple[float, float], wall1: str,
                          pos2: Tuple[float, float], wall2: str) -> float:
        """
        Calculate distance along the walls between two positions.
        This is the perimeter distance, not straight-line.
        """
        # Get indices
        idx1 = self.position_to_index.get((pos1, wall1))
        idx2 = self.position_to_index.get((pos2, wall2))
        
        if idx1 is None or idx2 is None:
            return float('inf')
        
        n = len(self.wall_positions)
        
        # Distance going forward
        forward = (idx2 - idx1) % n
        # Distance going backward
        backward = (idx1 - idx2) % n
        
        return min(forward, backward) * self.step_size
    
    def get_neighbors(self, state: CrawlerState) -> List[CrawlerState]:
        """
        Get reachable neighbor states.
        
        The robot can only move to adjacent positions on the wall,
        and must alternate feet.
        """
        neighbors = []
        current_pos = state.position
        current_wall = state.wall
        holding_foot = state.holding_foot
        
        # Find current index in wall positions
        current_idx = None
        for i, (pos, wall) in enumerate(self.wall_positions):
            if self.get_distance(pos, current_pos) < 0.01 and wall == current_wall:
                current_idx = i
                break
        
        if current_idx is None:
            return neighbors
        
        n = len(self.wall_positions)
        
        # Can move to positions within foot reach (1-2 steps typically)
        max_steps = int(self.foot_reach / self.step_size) + 1
        
        for direction in [-1, 1]:  # backward and forward along wall
            for steps in range(1, max_steps + 1):
                next_idx = (current_idx + direction * steps) % n
                next_pos, next_wall = self.wall_positions[next_idx]
                
                # Check if within foot reach
                if self.get_distance(current_pos, next_pos) <= self.foot_reach:
                    # Moving foot becomes the new holding foot
                    new_holding = 1 - holding_foot
                    neighbors.append(CrawlerState(next_pos, next_wall, new_holding))
        
        return neighbors
    
    def heuristic(self, state: CrawlerState, goal_pos: Tuple[float, float], goal_wall: str) -> float:
        """Heuristic: wall distance to goal"""
        return self.get_wall_distance(state.position, state.wall, goal_pos, goal_wall)
    
    def find_nearest_wall_position(self, pos: Tuple[float, float]) -> Tuple[Tuple[float, float], str]:
        """Find the nearest valid wall position to a given point"""
        min_dist = float('inf')
        nearest = self.wall_positions[0]
        
        for wall_pos, wall in self.wall_positions:
            dist = self.get_distance(pos, wall_pos)
            if dist < min_dist:
                min_dist = dist
                nearest = (wall_pos, wall)
        
        return nearest
    
    def find_path_astar(self, 
                        start_pos: Tuple[float, float],
                        goal_pos: Tuple[float, float],
                        start_foot: int = 0) -> Optional[List[CrawlerState]]:
        """
        Find optimal path using A* algorithm.
        
        Args:
            start_pos: Starting position (will snap to nearest wall position)
            goal_pos: Goal position (will snap to nearest wall position)
            start_foot: Which foot is initially holding (0 or 1)
            
        Returns:
            List of states representing the path, or None if no path found
        """
        # Snap to nearest wall positions
        start_wall_pos, start_wall = self.find_nearest_wall_position(start_pos)
        goal_wall_pos, goal_wall = self.find_nearest_wall_position(goal_pos)
        
        print(f"Start snapped to: {start_wall_pos} on {start_wall} wall")
        print(f"Goal snapped to: {goal_wall_pos} on {goal_wall} wall")
        
        start_state = CrawlerState(start_wall_pos, start_wall, start_foot)
        
        # Priority queue
        open_set = []
        start_node = PriorityNode(
            f_cost=self.heuristic(start_state, goal_wall_pos, goal_wall),
            state=start_state,
            g_cost=0.0,
            parent=None
        )
        heapq.heappush(open_set, start_node)
        
        # Tracking
        closed_set: Set[CrawlerState] = set()
        g_costs: Dict[CrawlerState, float] = {start_state: 0.0}
        
        iterations = 0
        max_iterations = 10000
        
        while open_set and iterations < max_iterations:
            iterations += 1
            current_node = heapq.heappop(open_set)
            current_state = current_node.state
            
            # Check if reached goal
            if self.get_distance(current_state.position, goal_wall_pos) < 0.01:
                # Reconstruct path
                path = []
                node = current_node
                while node is not None:
                    path.append(node.state)
                    node = node.parent
                path.reverse()
                print(f"A* found path with {len(path)} steps in {iterations} iterations")
                return path
            
            if current_state in closed_set:
                continue
            closed_set.add(current_state)
            
            # Explore neighbors
            for next_state in self.get_neighbors(current_state):
                if next_state in closed_set:
                    continue
                
                # Movement cost
                move_cost = self.get_distance(current_state.position, next_state.position)
                tentative_g = current_node.g_cost + move_cost
                
                if next_state in g_costs and tentative_g >= g_costs[next_state]:
                    continue
                
                g_costs[next_state] = tentative_g
                f_cost = tentative_g + self.heuristic(next_state, goal_wall_pos, goal_wall)
                
                next_node = PriorityNode(
                    f_cost=f_cost,
                    state=next_state,
                    g_cost=tentative_g,
                    parent=current_node
                )
                heapq.heappush(open_set, next_node)
        
        print(f"A* failed after {iterations} iterations")
        return None


class WallCrawlerVisualizer:
    """2D Visualization for the wall-crawling robot"""
    
    def __init__(self, planner: WallCrawlerPlanner):
        self.planner = planner
        self.fig = None
        self.ax = None
        
        # Colors
        self.colors = {
            'wall': '#2C3E50',
            'wall_path': '#95A5A6',
            'floor': '#ECF0F1',
            'body': '#E74C3C',
            'foot_holding': '#27AE60',
            'foot_moving': '#9B59B6',
            'path': '#3498DB',
            'start': '#2ECC71',
            'goal': '#E74C3C',
            'foot_trail': '#BDC3C7'
        }
        
        self.wall_thickness = 0.08
    
    def setup_plot(self, title="Wall-Crawler A* Path Planning"):
        """Setup the matplotlib figure"""
        self.fig, self.ax = plt.subplots(figsize=(10, 10))
        
        margin = 0.3
        self.ax.set_xlim(-margin, self.planner.width + margin)
        self.ax.set_ylim(-margin, self.planner.height + margin)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel('X (meters)', fontsize=12)
        self.ax.set_ylabel('Y (meters)', fontsize=12)
        self.ax.set_title(title, fontsize=14, fontweight='bold')
        
    def draw_container(self):
        """Draw the container box with walls"""
        w = self.planner.width
        h = self.planner.height
        t = self.wall_thickness
        
        # Floor (inside of container)
        floor = patches.Rectangle(
            (0, 0), w, h,
            facecolor=self.colors['floor'],
            edgecolor='none'
        )
        self.ax.add_patch(floor)
        
        # Walls (as thick rectangles)
        # Bottom wall
        self.ax.add_patch(patches.Rectangle(
            (-t, -t), w + 2*t, t,
            facecolor=self.colors['wall'], edgecolor='black', linewidth=2
        ))
        # Top wall
        self.ax.add_patch(patches.Rectangle(
            (-t, h), w + 2*t, t,
            facecolor=self.colors['wall'], edgecolor='black', linewidth=2
        ))
        # Left wall
        self.ax.add_patch(patches.Rectangle(
            (-t, 0), t, h,
            facecolor=self.colors['wall'], edgecolor='black', linewidth=2
        ))
        # Right wall
        self.ax.add_patch(patches.Rectangle(
            (w, 0), t, h,
            facecolor=self.colors['wall'], edgecolor='black', linewidth=2
        ))
        
        # Draw discrete positions on walls
        for (pos, wall) in self.planner.wall_positions:
            self.ax.plot(pos[0], pos[1], 'o', color=self.colors['wall_path'], 
                        markersize=4, alpha=0.5)
    
    def draw_robot_state(self, state: CrawlerState, prev_state: CrawlerState = None,
                         step_num: int = None, total_steps: int = None):
        """
        Draw the robot at a given state.
        Shows body, both feet, and which foot is holding vs moving.
        """
        pos = state.position
        holding = state.holding_foot
        
        # Determine foot positions based on wall and movement
        # The holding foot is at the current position
        # The other foot is slightly offset (simulating alternating steps)
        
        foot_offset = 0.06  # Visual offset for feet
        
        # Get wall normal direction for offsetting feet
        wall = state.wall
        if wall == 'bottom':
            normal = (0, 1)
            tangent = (1, 0)
        elif wall == 'right':
            normal = (-1, 0)
            tangent = (0, 1)
        elif wall == 'top':
            normal = (0, -1)
            tangent = (-1, 0)
        else:  # left
            normal = (1, 0)
            tangent = (0, -1)
        
        # Body position (slightly off the wall)
        body_offset = 0.08
        body_pos = (pos[0] + normal[0] * body_offset, 
                    pos[1] + normal[1] * body_offset)
        
        # Foot positions
        foot1_pos = (pos[0] - tangent[0] * foot_offset,
                     pos[1] - tangent[1] * foot_offset)
        foot2_pos = (pos[0] + tangent[0] * foot_offset,
                     pos[1] + tangent[1] * foot_offset)
        
        # Determine which foot is holding and which is moving
        if holding == 0:
            holding_pos = foot1_pos
            moving_pos = foot2_pos
        else:
            holding_pos = foot2_pos
            moving_pos = foot1_pos
        
        # Draw connection to wall (feet gripping the wall)
        self.ax.plot([holding_pos[0], holding_pos[0]], 
                    [holding_pos[1], holding_pos[1]], 
                    'o', color=self.colors['foot_holding'], markersize=12, zorder=15)
        self.ax.plot([moving_pos[0], moving_pos[0]], 
                    [moving_pos[1], moving_pos[1]], 
                    'o', color=self.colors['foot_moving'], markersize=10, zorder=15)
        
        # Draw body
        body = patches.Circle(
            body_pos, 0.05,
            facecolor=self.colors['body'],
            edgecolor='black',
            linewidth=2,
            zorder=20
        )
        self.ax.add_patch(body)
        
        # Draw legs connecting body to feet
        self.ax.plot([body_pos[0], holding_pos[0]], [body_pos[1], holding_pos[1]],
                    color=self.colors['foot_holding'], linewidth=3, zorder=10)
        self.ax.plot([body_pos[0], moving_pos[0]], [body_pos[1], moving_pos[1]],
                    color=self.colors['foot_moving'], linewidth=3, zorder=10)
        
        # If we have previous state, show the "step" movement
        if prev_state is not None:
            prev_pos = prev_state.position
            # Draw arrow showing movement
            self.ax.annotate('', xy=pos, xytext=prev_pos,
                           arrowprops=dict(arrowstyle='->', color=self.colors['path'],
                                         lw=2, alpha=0.7))
        
        return body_pos
    
    def draw_path_line(self, path: List[CrawlerState]):
        """Draw the planned path as a line on the wall"""
        if not path:
            return
        
        xs = [state.position[0] for state in path]
        ys = [state.position[1] for state in path]
        
        # Draw path line
        self.ax.plot(xs, ys, color=self.colors['path'], linewidth=2, 
                    linestyle='-', alpha=0.6, zorder=5)
        
        # Mark start and goal
        self.ax.scatter([xs[0]], [ys[0]], color=self.colors['start'], 
                       s=200, marker='*', zorder=25, label='Start', edgecolor='black')
        self.ax.scatter([xs[-1]], [ys[-1]], color=self.colors['goal'], 
                       s=200, marker='*', zorder=25, label='Goal', edgecolor='black')
    
    def animate_path(self, path: List[CrawlerState], interval: int = 800,
                     save_file: str = None):
        """
        Create step-by-step animation of robot crawling along the wall.
        """
        if not path:
            print("No path to animate!")
            return
        
        self.setup_plot()
        self.draw_container()
        self.draw_path_line(path)
        
        # Store elements to clear each frame
        self.robot_elements = []
        self.step_text = None
        
        # Legend (add once at start)
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor=self.colors['foot_holding'],
                   markersize=12, label='Holding Foot'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor=self.colors['foot_moving'],
                   markersize=10, label='Moving Foot'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor=self.colors['start'],
                   markersize=15, label='Start'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor=self.colors['goal'],
                   markersize=15, label='Goal'),
        ]
        self.ax.legend(handles=legend_elements, loc='upper left', fontsize=10)
        
        plt.ion()  # Enable interactive mode
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.5)
        
        # Manual frame-by-frame animation for better control
        for frame in range(len(path)):
            # Clear previous robot drawing
            for elem in self.robot_elements:
                try:
                    elem.remove()
                except:
                    pass
            self.robot_elements = []
            
            state = path[frame]
            prev_state = path[frame - 1] if frame > 0 else None
            
            # Draw current state
            pos = state.position
            holding = state.holding_foot
            wall = state.wall
            
            # Calculate positions based on wall
            if wall == 'bottom':
                normal = (0, 1)
                tangent = (1, 0)
            elif wall == 'right':
                normal = (-1, 0)
                tangent = (0, 1)
            elif wall == 'top':
                normal = (0, -1)
                tangent = (-1, 0)
            else:  # left
                normal = (1, 0)
                tangent = (0, -1)
            
            body_offset = 0.1
            foot_offset = 0.05
            
            body_pos = (pos[0] + normal[0] * body_offset,
                       pos[1] + normal[1] * body_offset)
            
            foot1_pos = (pos[0] - tangent[0] * foot_offset,
                        pos[1] - tangent[1] * foot_offset)
            foot2_pos = (pos[0] + tangent[0] * foot_offset,
                        pos[1] + tangent[1] * foot_offset)
            
            if holding == 0:
                holding_pos, moving_pos = foot1_pos, foot2_pos
            else:
                holding_pos, moving_pos = foot2_pos, foot1_pos
            
            # Draw feet
            hold_circle = patches.Circle(holding_pos, 0.04,
                                         facecolor=self.colors['foot_holding'],
                                         edgecolor='black', linewidth=2, zorder=15)
            move_circle = patches.Circle(moving_pos, 0.035,
                                         facecolor=self.colors['foot_moving'],
                                         edgecolor='black', linewidth=1.5, zorder=15)
            self.ax.add_patch(hold_circle)
            self.ax.add_patch(move_circle)
            self.robot_elements.extend([hold_circle, move_circle])
            
            # Draw body
            body = patches.Ellipse(body_pos, 0.12, 0.08,
                                   facecolor=self.colors['body'],
                                   edgecolor='black', linewidth=2, zorder=20)
            self.ax.add_patch(body)
            self.robot_elements.append(body)
            
            # Draw legs
            leg1, = self.ax.plot([body_pos[0], holding_pos[0]], 
                                [body_pos[1], holding_pos[1]],
                                color=self.colors['foot_holding'], linewidth=4, zorder=10)
            leg2, = self.ax.plot([body_pos[0], moving_pos[0]], 
                                [body_pos[1], moving_pos[1]],
                                color=self.colors['foot_moving'], linewidth=3, zorder=10)
            self.robot_elements.extend([leg1, leg2])
            
            # Show previous position (ghost)
            if prev_state is not None:
                ghost = patches.Circle(prev_state.position, 0.03,
                                       facecolor='gray', alpha=0.3, zorder=5)
                self.ax.add_patch(ghost)
                self.robot_elements.append(ghost)
            
            # Update title with step info
            foot_name = "Foot 1" if holding == 0 else "Foot 2"
            self.ax.set_title(
                f"Wall-Crawler A* Path Planning\n"
                f"Step {frame + 1}/{len(path)} | Wall: {wall.capitalize()} | "
                f"Holding: {foot_name}\n"
                f"Position: ({pos[0]:.2f}, {pos[1]:.2f})",
                fontsize=12, fontweight='bold'
            )
            
            # Force redraw
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
            plt.pause(interval / 1000.0)  # Convert ms to seconds
        
        print("Animation complete!")
        
        # Save as GIF if requested
        if save_file:
            print(f"Saving animation to {save_file}...")
            # Recreate animation for saving
            self.setup_plot()
            self.draw_container()
            self.draw_path_line(path)
            self.robot_elements = []
            
            def update_for_save(frame):
                for elem in self.robot_elements:
                    try:
                        elem.remove()
                    except:
                        pass
                self.robot_elements = []
                
                state = path[frame]
                pos = state.position
                holding = state.holding_foot
                wall = state.wall
                
                if wall == 'bottom':
                    normal = (0, 1)
                    tangent = (1, 0)
                elif wall == 'right':
                    normal = (-1, 0)
                    tangent = (0, 1)
                elif wall == 'top':
                    normal = (0, -1)
                    tangent = (-1, 0)
                else:
                    normal = (1, 0)
                    tangent = (0, -1)
                
                body_offset = 0.1
                foot_offset = 0.05
                body_pos = (pos[0] + normal[0] * body_offset,
                           pos[1] + normal[1] * body_offset)
                foot1_pos = (pos[0] - tangent[0] * foot_offset,
                            pos[1] - tangent[1] * foot_offset)
                foot2_pos = (pos[0] + tangent[0] * foot_offset,
                            pos[1] + tangent[1] * foot_offset)
                
                if holding == 0:
                    holding_pos, moving_pos = foot1_pos, foot2_pos
                else:
                    holding_pos, moving_pos = foot2_pos, foot1_pos
                
                hold_circle = patches.Circle(holding_pos, 0.04,
                                             facecolor=self.colors['foot_holding'],
                                             edgecolor='black', linewidth=2, zorder=15)
                move_circle = patches.Circle(moving_pos, 0.035,
                                             facecolor=self.colors['foot_moving'],
                                             edgecolor='black', linewidth=1.5, zorder=15)
                self.ax.add_patch(hold_circle)
                self.ax.add_patch(move_circle)
                self.robot_elements.extend([hold_circle, move_circle])
                
                body = patches.Ellipse(body_pos, 0.12, 0.08,
                                       facecolor=self.colors['body'],
                                       edgecolor='black', linewidth=2, zorder=20)
                self.ax.add_patch(body)
                self.robot_elements.append(body)
                
                leg1, = self.ax.plot([body_pos[0], holding_pos[0]], 
                                    [body_pos[1], holding_pos[1]],
                                    color=self.colors['foot_holding'], linewidth=4, zorder=10)
                leg2, = self.ax.plot([body_pos[0], moving_pos[0]], 
                                    [body_pos[1], moving_pos[1]],
                                    color=self.colors['foot_moving'], linewidth=3, zorder=10)
                self.robot_elements.extend([leg1, leg2])
                
                foot_name = "Foot 1" if holding == 0 else "Foot 2"
                self.ax.set_title(
                    f"Wall-Crawler A* Path Planning\n"
                    f"Step {frame + 1}/{len(path)} | Wall: {wall.capitalize()} | "
                    f"Holding: {foot_name}",
                    fontsize=12, fontweight='bold'
                )
                
                return self.robot_elements
            
            anim = FuncAnimation(self.fig, update_for_save, frames=len(path),
                               interval=interval, blit=False, repeat=True)
            anim.save(save_file, writer='pillow', fps=1.5)
            print("Animation saved!")
        
        # Keep window open
        plt.ioff()
        plt.show()
    
    def plot_static(self, path: List[CrawlerState], save_file: str = None):
        """Create static plot showing all steps"""
        self.setup_plot("Wall-Crawler Path - All Steps")
        self.draw_container()
        self.draw_path_line(path)
        
        # Draw all robot positions with transparency
        for i, state in enumerate(path):
            alpha = 0.3 + 0.7 * (i / len(path))
            pos = state.position
            self.ax.plot(pos[0], pos[1], 'o', color=self.colors['body'],
                        markersize=8, alpha=alpha)
            # Label steps
            self.ax.annotate(str(i+1), (pos[0], pos[1]), 
                           textcoords="offset points", xytext=(5, 5),
                           fontsize=8, alpha=alpha)
        
        # Legend
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='*', color='w', markerfacecolor=self.colors['start'],
                   markersize=15, label='Start'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor=self.colors['goal'],
                   markersize=15, label='Goal'),
            Line2D([0], [0], color=self.colors['path'], linewidth=2, label='Path'),
        ]
        self.ax.legend(handles=legend_elements, loc='upper left')
        
        if save_file:
            plt.savefig(save_file, dpi=150, bbox_inches='tight')
            print(f"Static plot saved to {save_file}")
        
        plt.show()


def main():
    """Main function to run the wall-crawler A* planner demo"""
    
    print("=" * 60)
    print("Wall-Crawler Bipedal Robot A* Path Planner")
    print("=" * 60)
    print("\nThe robot moves ONLY along the walls of the container,")
    print("like a snail or wall-crawling bipedal with static stability.")
    print("=" * 60)
    
    # Create planner - 2m x 2m container
    planner = WallCrawlerPlanner(
        container_size=(2.0, 2.0),
        step_size=0.15,      # Discrete positions every 15cm
        foot_reach=0.25      # Each foot can reach 25cm
    )
    
    print(f"\nContainer size: {planner.width}m x {planner.height}m")
    print(f"Step size: {planner.step_size}m")
    print(f"Foot reach: {planner.foot_reach}m")
    print(f"Total perimeter positions: {len(planner.wall_positions)}")
    
    # Define start and goal positions
    # Start: bottom-left corner area
    # Goal: top-right corner area (must go around the walls!)
    start = (0.2, 0.0)   # Near bottom-left on bottom wall
    goal = (2.0, 1.8)    # Near top-right on right wall
    
    print(f"\nStart position: {start}")
    print(f"Goal position: {goal}")
    
    # Find path using A*
    print("\nRunning A* search...")
    start_time = time.time()
    path = planner.find_path_astar(start, goal, start_foot=0)
    elapsed = time.time() - start_time
    
    if path:
        print(f"\nPath found in {elapsed:.3f} seconds!")
        print(f"Path length: {len(path)} steps")
        print("\nPath steps:")
        for i, state in enumerate(path):
            foot = "Foot 1" if state.holding_foot == 0 else "Foot 2"
            print(f"  Step {i+1:2d}: Wall={state.wall:6s} | "
                  f"Pos=({state.position[0]:.2f}, {state.position[1]:.2f}) | "
                  f"Holding: {foot}")
    else:
        print("No path found!")
        return
    
    # Create visualizer
    viz = WallCrawlerVisualizer(planner)
    
    # Static plot first
    print("\nGenerating static visualization...")
    viz.plot_static(path, save_file='wall_crawler_path_static.png')
    
    # Animate
    print("\nGenerating step-by-step animation...")
    viz.animate_path(path, interval=800, save_file='wall_crawler_animation.gif')


if __name__ == "__main__":
    main()
