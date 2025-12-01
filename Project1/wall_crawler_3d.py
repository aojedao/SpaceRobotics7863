#!/usr/bin/env python3
"""
3D Wall-Crawler Bipedal Robot A* Path Planner

This script simulates a bipedal wall-crawling robot in a 3D environment.
The robot moves ONLY along the inner walls of a 3D box container.

The robot has two "feet" and moves with static stability:
- One foot stays fixed on the wall
- The other foot moves to the next position
- Then switch roles

Movement is constrained to the walls (all 6 faces) of the container.
"""

import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection
import heapq
from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Set, Dict
import time


@dataclass
class CrawlerState3D:
    """State representation for the 3D wall-crawler robot"""
    position: Tuple[float, float, float]  # (x, y, z) position on wall
    wall: str  # Which wall: 'floor', 'ceiling', 'front', 'back', 'left', 'right'
    holding_foot: int  # Which foot is currently holding (0 or 1)
    
    def __hash__(self):
        # Round position for hashing
        rounded_pos = (round(self.position[0], 3), 
                      round(self.position[1], 3), 
                      round(self.position[2], 3))
        return hash((rounded_pos, self.wall, self.holding_foot))
    
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
    state: CrawlerState3D = field(compare=False)
    g_cost: float = field(compare=False)
    parent: Optional['PriorityNode'] = field(compare=False, default=None)


class WallCrawlerPlanner3D:
    """
    3D Wall-crawling bipedal robot planner.
    
    The robot moves only along the walls (6 faces) of a 3D box container.
    It uses two feet for static stability - one holds while the other moves.
    """
    
    def __init__(self, 
                 container_size: Tuple[float, float, float] = (2.0, 2.0, 1.5),
                 step_size: float = 0.2,
                 foot_reach: float = 0.35):
        """
        Initialize the 3D wall crawler planner.
        
        Args:
            container_size: (width_x, depth_y, height_z) of the container
            step_size: Distance between discrete positions on the walls
            foot_reach: Maximum reach of each foot
        """
        self.width, self.depth, self.height = container_size
        self.step_size = step_size
        self.foot_reach = foot_reach
        
        # Create wall positions (discrete points on all 6 faces)
        self.wall_positions = self._create_wall_positions()
        
        # Create adjacency for quick neighbor lookup
        self.position_to_index = {}
        for i, (pos, wall) in enumerate(self.wall_positions):
            key = (round(pos[0], 3), round(pos[1], 3), round(pos[2], 3), wall)
            self.position_to_index[key] = i
        
        print(f"Created {len(self.wall_positions)} positions on all walls")
        
    def _create_wall_positions(self) -> List[Tuple[Tuple[float, float, float], str]]:
        """Create discrete positions on all 6 walls of the container"""
        positions = []
        
        # Generate grid points
        xs = np.arange(0, self.width + self.step_size/2, self.step_size)
        ys = np.arange(0, self.depth + self.step_size/2, self.step_size)
        zs = np.arange(0, self.height + self.step_size/2, self.step_size)
        
        # Floor (z = 0)
        for x in xs:
            for y in ys:
                positions.append(((x, y, 0.0), 'floor'))
        
        # Ceiling (z = height)
        for x in xs:
            for y in ys:
                positions.append(((x, y, self.height), 'ceiling'))
        
        # Front wall (y = 0)
        for x in xs:
            for z in zs[1:-1]:  # Skip edges already covered
                positions.append(((x, 0.0, z), 'front'))
        
        # Back wall (y = depth)
        for x in xs:
            for z in zs[1:-1]:
                positions.append(((x, self.depth, z), 'back'))
        
        # Left wall (x = 0)
        for y in ys[1:-1]:  # Skip edges
            for z in zs[1:-1]:
                positions.append(((0.0, y, z), 'left'))
        
        # Right wall (x = width)
        for y in ys[1:-1]:
            for z in zs[1:-1]:
                positions.append(((self.width, y, z), 'right'))
        
        return positions
    
    def get_distance(self, pos1: Tuple[float, float, float], 
                     pos2: Tuple[float, float, float]) -> float:
        """Euclidean distance between two 3D positions"""
        return np.sqrt((pos1[0] - pos2[0])**2 + 
                      (pos1[1] - pos2[1])**2 + 
                      (pos1[2] - pos2[2])**2)
    
    def get_wall_normal(self, wall: str) -> Tuple[float, float, float]:
        """Get the inward-pointing normal vector for a wall"""
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
        """Check if robot can transition between two walls (must be adjacent)"""
        # All walls can potentially connect at edges
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
        """Get reachable neighbor states"""
        neighbors = []
        current_pos = state.position
        holding_foot = state.holding_foot
        
        for target_pos, target_wall in self.wall_positions:
            # Skip same position
            if target_pos == current_pos:
                continue
            
            # Check if within foot reach
            dist = self.get_distance(current_pos, target_pos)
            if dist > self.foot_reach:
                continue
            
            # Check wall adjacency
            if not self.can_transition(state.wall, target_wall):
                continue
            
            # Valid move - switch holding foot
            new_holding = 1 - holding_foot
            neighbors.append(CrawlerState3D(target_pos, target_wall, new_holding))
        
        return neighbors
    
    def heuristic(self, pos1: Tuple[float, float, float], 
                  pos2: Tuple[float, float, float]) -> float:
        """Heuristic: Euclidean distance"""
        return self.get_distance(pos1, pos2)
    
    def find_nearest_wall_position(self, pos: Tuple[float, float, float]) -> Tuple[Tuple[float, float, float], str]:
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
                        start_pos: Tuple[float, float, float],
                        goal_pos: Tuple[float, float, float],
                        start_foot: int = 0) -> Optional[List[CrawlerState3D]]:
        """Find optimal path using A* algorithm"""
        
        # Snap to nearest wall positions
        start_wall_pos, start_wall = self.find_nearest_wall_position(start_pos)
        goal_wall_pos, goal_wall = self.find_nearest_wall_position(goal_pos)
        
        print(f"Start snapped to: {start_wall_pos} on {start_wall}")
        print(f"Goal snapped to: {goal_wall_pos} on {goal_wall}")
        
        start_state = CrawlerState3D(start_wall_pos, start_wall, start_foot)
        
        # Priority queue
        open_set = []
        start_node = PriorityNode(
            f_cost=self.heuristic(start_wall_pos, goal_wall_pos),
            state=start_state,
            g_cost=0.0,
            parent=None
        )
        heapq.heappush(open_set, start_node)
        
        # Tracking
        closed_set: Set[CrawlerState3D] = set()
        g_costs: Dict[CrawlerState3D, float] = {start_state: 0.0}
        
        iterations = 0
        max_iterations = 50000
        
        while open_set and iterations < max_iterations:
            iterations += 1
            current_node = heapq.heappop(open_set)
            current_state = current_node.state
            
            # Check if reached goal
            if self.get_distance(current_state.position, goal_wall_pos) < 0.01:
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


class WallCrawlerVisualizer3D:
    """3D Visualization for the wall-crawling robot"""
    
    def __init__(self, planner: WallCrawlerPlanner3D):
        self.planner = planner
        self.fig = None
        self.ax = None
        
        # Colors
        self.colors = {
            'floor': '#D4E6F1',
            'ceiling': '#FCF3CF',
            'walls': '#E8DAEF',
            'body': '#E74C3C',
            'foot_holding': '#27AE60',
            'foot_moving': '#9B59B6',
            'path': '#3498DB',
            'start': '#2ECC71',
            'goal': '#E74C3C',
        }
        
        self.robot_elements = []
    
    def setup_plot(self, title="3D Wall-Crawler A* Path Planning"):
        """Setup the 3D matplotlib figure"""
        self.fig = plt.figure(figsize=(12, 10))
        self.ax = self.fig.add_subplot(111, projection='3d')
        
        # Set axis limits with margin
        margin = 0.3
        self.ax.set_xlim(-margin, self.planner.width + margin)
        self.ax.set_ylim(-margin, self.planner.depth + margin)
        self.ax.set_zlim(-margin, self.planner.height + margin)
        
        self.ax.set_xlabel('X (meters)', fontsize=10)
        self.ax.set_ylabel('Y (meters)', fontsize=10)
        self.ax.set_zlabel('Z (meters)', fontsize=10)
        self.ax.set_title(title, fontsize=14, fontweight='bold')
        
        # Set viewing angle
        self.ax.view_init(elev=25, azim=45)
    
    def draw_container(self, alpha=0.3):
        """Draw the 3D container box"""
        w, d, h = self.planner.width, self.planner.depth, self.planner.height
        
        # Define the 6 faces of the box
        faces = [
            # Floor (z=0)
            [[0, 0, 0], [w, 0, 0], [w, d, 0], [0, d, 0]],
            # Ceiling (z=h)
            [[0, 0, h], [w, 0, h], [w, d, h], [0, d, h]],
            # Front (y=0)
            [[0, 0, 0], [w, 0, 0], [w, 0, h], [0, 0, h]],
            # Back (y=d)
            [[0, d, 0], [w, d, 0], [w, d, h], [0, d, h]],
            # Left (x=0)
            [[0, 0, 0], [0, d, 0], [0, d, h], [0, 0, h]],
            # Right (x=w)
            [[w, 0, 0], [w, d, 0], [w, d, h], [w, 0, h]],
        ]
        
        face_colors = [
            self.colors['floor'],
            self.colors['ceiling'],
            self.colors['walls'],
            self.colors['walls'],
            self.colors['walls'],
            self.colors['walls'],
        ]
        
        for face, color in zip(faces, face_colors):
            poly = Poly3DCollection([face], alpha=alpha, facecolor=color, 
                                   edgecolor='black', linewidth=1)
            self.ax.add_collection3d(poly)
        
        # Draw grid points on walls
        for (pos, wall) in self.planner.wall_positions:
            self.ax.scatter(*pos, c='gray', s=5, alpha=0.3)
    
    def draw_path_line(self, path: List[CrawlerState3D]):
        """Draw the planned path as a 3D line"""
        if not path:
            return
        
        xs = [state.position[0] for state in path]
        ys = [state.position[1] for state in path]
        zs = [state.position[2] for state in path]
        
        # Draw path line
        self.ax.plot(xs, ys, zs, color=self.colors['path'], linewidth=2, 
                    linestyle='-', alpha=0.7)
        
        # Mark start and goal
        self.ax.scatter([xs[0]], [ys[0]], [zs[0]], c=self.colors['start'], 
                       s=200, marker='*', label='Start')
        self.ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], c=self.colors['goal'], 
                       s=200, marker='*', label='Goal')
    
    def draw_robot(self, state: CrawlerState3D, prev_state: CrawlerState3D = None):
        """Draw the robot at current state with proper bipedal stance"""
        pos = state.position
        wall = state.wall
        holding = state.holding_foot
        
        # Get wall normal for body offset
        normal = self.planner.get_wall_normal(wall)
        body_offset = 0.12
        
        # Calculate tangent directions for foot placement along the wall
        if wall in ['floor', 'ceiling']:
            tangent1 = np.array([1, 0, 0])
            tangent2 = np.array([0, 1, 0])
        elif wall in ['front', 'back']:
            tangent1 = np.array([1, 0, 0])
            tangent2 = np.array([0, 0, 1])
        else:  # left, right
            tangent1 = np.array([0, 1, 0])
            tangent2 = np.array([0, 0, 1])
        
        # Current position as numpy array
        pos_arr = np.array(pos)
        normal_arr = np.array(normal)
        
        # BIPEDAL LOCOMOTION: One foot stays still, other foot moves
        # holding_foot indicates which foot is currently gripping (holding weight)
        # The OTHER foot is the one that just moved to the new position
        if prev_state is not None:
            prev_pos = np.array(prev_state.position)
            
            # The holding foot (from prev state) stays at previous position
            # The moving foot lands at the new position
            if holding == 0:
                # Foot 0 is now holding at new position (just stepped here)
                # Foot 1 is still at the previous position (anchor)
                foot0_pos = pos_arr  # Foot 0 just landed here
                foot1_pos = prev_pos  # Foot 1 stayed still (anchor)
            else:
                # Foot 1 is now holding at new position (just stepped here)
                # Foot 0 is still at the previous position (anchor)
                foot0_pos = prev_pos  # Foot 0 stayed still (anchor)
                foot1_pos = pos_arr  # Foot 1 just landed here
            
            # Holding foot = the one currently gripping (just stepped)
            # Moving foot = the one still at previous position (will move next)
            if holding == 0:
                holding_pos = foot0_pos
                moving_pos = foot1_pos
            else:
                holding_pos = foot1_pos
                moving_pos = foot0_pos
        else:
            # First frame - feet side by side
            foot_offset = 0.06
            if holding == 0:
                holding_pos = pos_arr - tangent1 * foot_offset
                moving_pos = pos_arr + tangent1 * foot_offset
            else:
                holding_pos = pos_arr + tangent1 * foot_offset
                moving_pos = pos_arr - tangent1 * foot_offset
        
        # Body position (between feet, offset from wall)
        body_pos = (holding_pos + moving_pos) / 2 + normal_arr * body_offset
        
        elements = []
        
        # Draw feet as spheres on the wall
        hold_foot = self.ax.scatter(*holding_pos, c=self.colors['foot_holding'], 
                                    s=120, marker='o', edgecolor='black', linewidth=1.5,
                                    zorder=10, label='Holding Foot' if not hasattr(self, '_legend_added') else '')
        move_foot = self.ax.scatter(*moving_pos, c=self.colors['foot_moving'], 
                                    s=80, marker='o', edgecolor='black', linewidth=1,
                                    zorder=10, label='Moving Foot' if not hasattr(self, '_legend_added') else '')
        elements.extend([hold_foot, move_foot])
        
        # Draw body
        body = self.ax.scatter(*body_pos, c=self.colors['body'], s=180, 
                               marker='o', edgecolor='black', linewidth=2, zorder=15)
        elements.append(body)
        
        # Draw legs connecting body to feet
        leg1, = self.ax.plot([body_pos[0], holding_pos[0]], 
                            [body_pos[1], holding_pos[1]],
                            [body_pos[2], holding_pos[2]],
                            color=self.colors['foot_holding'], linewidth=4, zorder=5)
        leg2, = self.ax.plot([body_pos[0], moving_pos[0]], 
                            [body_pos[1], moving_pos[1]],
                            [body_pos[2], moving_pos[2]],
                            color=self.colors['foot_moving'], linewidth=3, zorder=5)
        elements.extend([leg1, leg2])
        
        # Draw small markers showing feet are gripping the wall
        grip1, = self.ax.plot([holding_pos[0]], [holding_pos[1]], [holding_pos[2]], 
                             'o', color='black', markersize=3, zorder=11)
        grip2, = self.ax.plot([moving_pos[0]], [moving_pos[1]], [moving_pos[2]], 
                             'o', color='gray', markersize=2, zorder=11)
        elements.extend([grip1, grip2])
        
        self._legend_added = True
        
        return elements
    
    def animate_path(self, path: List[CrawlerState3D], interval: float = 0.3,
                     save_file: str = None):
        """Create step-by-step 3D animation with proper bipedal motion"""
        if not path:
            print("No path to animate!")
            return
        
        # Fixed view angle for better visibility
        base_azim = 45
        base_elev = 25
        
        # Save GIF FIRST (before showing animation) so it's saved even if window is closed early
        if save_file:
            print(f"Saving animation to {save_file}...")
            self._save_animation_gif(path, save_file, base_azim)
            print("GIF saved! Now showing interactive animation...")
        
        self.setup_plot()
        self.draw_container(alpha=0.2)
        self.draw_path_line(path)
        
        plt.ion()
        plt.tight_layout()
        plt.show(block=False)
        plt.pause(0.3)
        
        robot_elements = []
        self._legend_added = False
        
        self.ax.view_init(elev=base_elev, azim=base_azim)
        
        try:
            for frame in range(len(path)):
                # Clear previous robot
                for elem in robot_elements:
                    try:
                        elem.remove()
                    except:
                        pass
                robot_elements = []
                
                state = path[frame]
                prev_state = path[frame - 1] if frame > 0 else None
                
                # Draw robot with previous state for proper foot positioning
                elements = self.draw_robot(state, prev_state)
                robot_elements.extend(elements)
                
                # Update title
                foot_name = "Foot 1" if state.holding_foot == 0 else "Foot 2"
                pos = state.position
                self.ax.set_title(
                    f"3D Wall-Crawler | Step {frame + 1}/{len(path)}\n"
                    f"Wall: {state.wall.capitalize()} | Holding: {foot_name} | "
                    f"Pos: ({pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f})",
                    fontsize=11, fontweight='bold'
                )
                
                # Fixed camera - no rotation
                
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
    
    def _save_animation_gif(self, path: List[CrawlerState3D], save_file: str, base_azim: float):
        """Save the animation as a GIF file by saving individual frames"""
        import imageio
        import tempfile
        import os
        
        # Fixed camera view
        fixed_elev = 25
        fixed_azim = 45
        
        frames = []
        temp_dir = tempfile.mkdtemp()
        
        try:
            for frame in range(len(path)):
                # Create fresh figure for each frame
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection='3d')
                
                # Set axis limits
                margin = 0.2
                ax.set_xlim(-margin, self.planner.width + margin)
                ax.set_ylim(-margin, self.planner.depth + margin)
                ax.set_zlim(-margin, self.planner.height + margin)
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')
                ax.set_zlabel('Z (m)')
                
                # Draw container
                w, d, h = self.planner.width, self.planner.depth, self.planner.height
                faces = [
                    [[0, 0, 0], [w, 0, 0], [w, d, 0], [0, d, 0]],
                    [[0, 0, h], [w, 0, h], [w, d, h], [0, d, h]],
                    [[0, 0, 0], [w, 0, 0], [w, 0, h], [0, 0, h]],
                    [[0, d, 0], [w, d, 0], [w, d, h], [0, d, h]],
                    [[0, 0, 0], [0, d, 0], [0, d, h], [0, 0, h]],
                    [[w, 0, 0], [w, d, 0], [w, d, h], [w, 0, h]],
                ]
                for face in faces:
                    poly = Poly3DCollection([face], alpha=0.15, facecolor='lightblue', 
                                           edgecolor='black', linewidth=0.5)
                    ax.add_collection3d(poly)
                
                # Draw path
                xs = [s.position[0] for s in path]
                ys = [s.position[1] for s in path]
                zs = [s.position[2] for s in path]
                ax.plot(xs, ys, zs, color='blue', linewidth=2, alpha=0.5)
                ax.scatter([xs[0]], [ys[0]], [zs[0]], c='green', s=100, marker='*')
                ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], c='red', s=100, marker='*')
                
                # Draw robot at current frame
                state = path[frame]
                prev_state = path[frame - 1] if frame > 0 else None
                pos = np.array(state.position)
                wall = state.wall
                holding = state.holding_foot
                
                normal = np.array(self.planner.get_wall_normal(wall))
                
                if wall in ['floor', 'ceiling']:
                    tangent1 = np.array([1, 0, 0])
                elif wall in ['front', 'back']:
                    tangent1 = np.array([1, 0, 0])
                else:
                    tangent1 = np.array([0, 1, 0])
                
                if prev_state is not None:
                    prev_pos = np.array(prev_state.position)
                    
                    # BIPEDAL: One foot at new position, one at previous position
                    if holding == 0:
                        # Foot 0 just stepped to new position, Foot 1 at previous
                        foot0_pos = pos  # Foot 0 just landed here
                        foot1_pos = prev_pos  # Foot 1 stayed still
                    else:
                        # Foot 1 just stepped to new position, Foot 0 at previous
                        foot0_pos = prev_pos  # Foot 0 stayed still
                        foot1_pos = pos  # Foot 1 just landed here
                    
                    if holding == 0:
                        holding_pos = foot0_pos
                        moving_pos = foot1_pos
                    else:
                        holding_pos = foot1_pos
                        moving_pos = foot0_pos
                else:
                    holding_pos = pos - tangent1 * 0.06
                    moving_pos = pos + tangent1 * 0.06
                
                body_pos = (holding_pos + moving_pos) / 2 + normal * 0.12
                
                # Draw robot parts
                ax.scatter(*holding_pos, c='green', s=80, marker='o', edgecolor='black')
                ax.scatter(*moving_pos, c='purple', s=60, marker='o', edgecolor='black')
                ax.scatter(*body_pos, c='red', s=120, marker='o', edgecolor='black')
                ax.plot([body_pos[0], holding_pos[0]], [body_pos[1], holding_pos[1]], 
                       [body_pos[2], holding_pos[2]], color='green', linewidth=3)
                ax.plot([body_pos[0], moving_pos[0]], [body_pos[1], moving_pos[1]], 
                       [body_pos[2], moving_pos[2]], color='purple', linewidth=2)
                
                foot_name = "Foot 1" if holding == 0 else "Foot 2"
                ax.set_title(f"3D Wall-Crawler | Step {frame + 1}/{len(path)}\n"
                            f"Wall: {wall.capitalize()} | Holding: {foot_name}",
                            fontsize=10, fontweight='bold')
                
                # Fixed camera view
                ax.view_init(elev=fixed_elev, azim=fixed_azim)
                
                # Save frame to temp file
                frame_path = os.path.join(temp_dir, f'frame_{frame:03d}.png')
                fig.savefig(frame_path, dpi=100, bbox_inches='tight')
                plt.close(fig)
                
                # Read frame for GIF
                frames.append(imageio.imread(frame_path))
            
            # Save as GIF
            imageio.mimsave(save_file, frames, duration=0.35)
            print(f"GIF saved to {save_file}")
            
        finally:
            # Cleanup temp files
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    def plot_static(self, path: List[CrawlerState3D], save_file: str = None):
        """Create static 3D plot showing the path"""
        self.setup_plot("3D Wall-Crawler Path Overview")
        self.draw_container(alpha=0.15)
        self.draw_path_line(path)
        
        # Draw robot positions along path with transparency
        for i, state in enumerate(path):
            alpha = 0.2 + 0.6 * (i / len(path))
            pos = state.position
            self.ax.scatter(*pos, c=self.colors['body'], s=30, alpha=alpha)
        
        # Draw final robot position
        if path:
            self._legend_added = False
            prev_state = path[-2] if len(path) > 1 else None
            self.draw_robot(path[-1], prev_state)
        
        self.ax.legend(loc='upper left')
        
        if save_file:
            plt.savefig(save_file, dpi=150, bbox_inches='tight')
            print(f"Static plot saved to {save_file}")
        
        plt.show()


def main():
    """Main function to run the 3D wall-crawler demo"""
    
    print("=" * 60)
    print("3D Wall-Crawler Bipedal Robot A* Path Planner")
    print("=" * 60)
    print("\nThe robot moves along the inner walls of a 3D box container,")
    print("like a wall-crawling bipedal with static stability.")
    print("=" * 60)
    
    # Create planner - 3D box container
    planner = WallCrawlerPlanner3D(
        container_size=(2.0, 2.0, 1.5),  # width, depth, height
        step_size=0.2,        # Discrete positions every 20cm
        foot_reach=0.35       # Each foot can reach 35cm
    )
    
    print(f"\nContainer size: {planner.width}m x {planner.depth}m x {planner.height}m")
    print(f"Step size: {planner.step_size}m")
    print(f"Foot reach: {planner.foot_reach}m")
    print(f"Total wall positions: {len(planner.wall_positions)}")
    
    # Fixed start position on floor
    start = (0.2, 0.2, 0.0)
    
    # Random goal position - pick a random wall position
    np.random.seed(int(time.time() * 1000) % 2**32)  # Different seed each run
    
    # Pick random goal from wall positions (excluding positions too close to start)
    valid_goals = [(pos, wall) for pos, wall in planner.wall_positions 
                   if planner.get_distance(pos, start) > 1.0]  # At least 1m away
    
    if valid_goals:
        goal_pos, goal_wall = valid_goals[np.random.randint(len(valid_goals))]
        goal = goal_pos
    else:
        goal = (1.8, 1.8, 1.5)  # Fallback
    
    print(f"\nStart position: {start} (floor)")
    print(f"Random goal position: ({goal[0]:.2f}, {goal[1]:.2f}, {goal[2]:.2f})")
    
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
            pos = state.position
            print(f"  Step {i+1:2d}: Wall={state.wall:7s} | "
                  f"Pos=({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) | "
                  f"Holding: {foot}")
    else:
        print("No path found!")
        return
    
    # Create visualizer
    viz = WallCrawlerVisualizer3D(planner)
    
    # Show static plot first
    print("\nGenerating static 3D visualization...")
    viz.plot_static(path, save_file='wall_crawler_3d_static.png')
    
    # Animate and save GIF
    print("\nGenerating step-by-step 3D animation...")
    viz.animate_path(path, interval=0.3, save_file='wall_crawler_3d_animation.gif')


if __name__ == "__main__":
    main()
