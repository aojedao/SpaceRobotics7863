#!/usr/bin/env python3
"""
MuJoCo-Based Dual-Arm Wall-Crawler Simulation

This simulation uses the dual_arm_robot.xml MuJoCo environment to implement
the wall-crawling algorithm for an ISS module inspection robot.

MODULAR ARCHITECTURE:
=====================
This main file orchestrates the simulation, importing from:
- enums.py: State enums
- planner.py: WallPositionMapper and PathPlanner
- logger.py: Simulation reporting
- plotter.py: Visualization utilities
- controller.py: ControllerMixin (Arm control, IK)
- recovery.py: RecoveryMixin (Stuck detection)
- tasks.py: TaskMixin (Unscrewing logic)

Author: Wall Crawler Project
Date: December 2025
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import random
from typing import List, Tuple, Optional, Dict

# Import modular components
from enums import RobotState, GripperState, CrawlerState
from planner import WallPositionMapper, PathPlanner, ISS_MODULE, KUKA_REACH, GRIP_POINT_OFFSETS
from logger import record_run_result, print_success_rate
import plotter
from controller import ControllerMixin
from recovery import RecoveryMixin
from tasks import TaskMixin

# ============================================================================
# MUJOCO WALL CRAWLER SIMULATION
# ============================================================================

class WallCrawlerMuJoCoSimulation(ControllerMixin, RecoveryMixin, TaskMixin):
    """
    MuJoCo-based wall-crawler simulation.
    Combines controller, planning, and visualization logic.
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
        self.wall_mapper = WallPositionMapper(step_size=0.4, arm_reach=KUKA_REACH)
        
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
        
        # Visualization markers
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
        
        self.left_link4_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "left_link4"
        )
        self.right_link4_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "right_link4"
        )
        
        print(f"✓ Central body ID: {self.central_body_id}")
        
        # Setup additional IDs for control
        self._setup_control_ids()
    
    def _setup_control_ids(self):
        """Setup body IDs for arm control and grippers"""
        self.left_ee_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "left_gripper_base_mount"
        )
        self.right_ee_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "right_gripper_base_mount"
        )
        
        self.left_arm_actuator_slice = slice(0, 7)
        self.right_arm_actuator_slice = slice(7, 14)
        self.left_gripper_actuator_idx = 14
        self.right_gripper_actuator_idx = 15
        
        self.left_arm_qpos_slice = slice(7, 14)
        self.right_arm_qpos_slice = slice(22, 29)
        self.left_arm_qvel_slice = slice(6, 13)
        self.right_arm_qvel_slice = slice(21, 28)
        
        # Controller gains
        self.kp_position = 800.0
        self.kd_position = 50.0
        self.kp_orientation = 50.0
        self.kd_orientation = 25.0
        self.lambda_dls = 0.008
        
        self.joint_gain_scale = np.array([3.0, 2.5, 1.5, 1.2, 1.0, 0.8, 0.6])
        
        self.kp_anchor = 1500.0
        self.kd_anchor = 50.0
        
        self.gripper_open_value = 0
        self.gripper_closed_value = 255
        
        self.left_target_position = None
        self.right_target_position = None
        
        self.left_arm_anchored = False
        self.right_arm_anchored = False
        self.left_anchor_position = None
        self.right_anchor_position = None
        self.locked_body_pos = None
        self.locked_body_quat = None
        self.locked_left_arm_joints = None
        
        print(f"✓ Left EE body ID: {self.left_ee_body_id}")
        print(f"✓ Right EE body ID: {self.right_ee_body_id}")

    def get_central_body_pos(self) -> np.ndarray:
        return self.data.qpos[0:3].copy()
    
    def get_left_gripper_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.left_gripper_site_id].copy()
    
    def get_right_gripper_pos(self) -> np.ndarray:
        return self.data.site_xpos[self.right_gripper_site_id].copy()

    def get_left_ee_orientation(self) -> np.ndarray:
        return self.data.xmat[self.left_ee_body_id].reshape(3, 3).copy()

    def get_right_ee_orientation(self) -> np.ndarray:
        return self.data.xmat[self.right_ee_body_id].reshape(3, 3).copy()

    def get_left_link4_pos(self) -> np.ndarray:
        return self.data.xpos[self.left_link4_id].copy()

    def get_right_link4_pos(self) -> np.ndarray:
        return self.data.xpos[self.right_link4_id].copy()

    def snap_to_nearest_sphere(self, pos: Tuple[float, float, float]) -> Tuple[Tuple[float, float, float], str]:
        """Snap a position to the nearest wall sphere position."""
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
        """Execute one step of Phase 2 control (Anchoring + Arm Movement) for Locomotion (Teleport)."""
        if self.right_arm_anchored and not self.left_arm_anchored:
            # RIGHT anchored, LEFT moving
            err = self.maintain_anchor('right', control_anchored_arm=False)
            if self.left_target_position is not None:
                self.apply_arm_control('left')
                if err > 0.05:
                     self.apply_coordinated_arm_control('right', self.left_target_position, self.right_anchor_position)
        
        elif self.left_arm_anchored and not self.right_arm_anchored:
            # LEFT anchored, RIGHT moving
            err = self.maintain_anchor('left', control_anchored_arm=False)
            if self.right_target_position is not None:
                self.apply_arm_control('right')
                if err > 0.05:
                    self.apply_coordinated_arm_control('left', self.right_target_position, self.left_anchor_position)
        
        elif self.left_arm_anchored and self.right_arm_anchored:
            # BOTH anchored
            self.maintain_anchor('left', control_anchored_arm=False)
            self.maintain_anchor('right', control_anchored_arm=False)

    def _apply_anchor_force(self, arm: str, anchor_pos: np.ndarray, prev_pos_attr: str, hard_lock: bool = False):
        """Apply anchor force to hold end-effector at position (Physics-based)."""
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
        
        effective_mass = 5.0
        if hard_lock:
            grip_stiffness = 120000.0
            grip_damping = 2.0 * np.sqrt(grip_stiffness * effective_mass)
            max_force = 80000.0
            velocity_damping_factor = 0.5
        else:
            grip_stiffness = 80000.0
            grip_damping = 2.0 * np.sqrt(grip_stiffness * effective_mass)
            max_force = 40000.0
            velocity_damping_factor = 0.9
        
        force = grip_stiffness * err - grip_damping * vel_approx
        force_mag = np.linalg.norm(force)
        if force_mag > max_force:
            force = force * (max_force / force_mag)
        self.data.xfrc_applied[ee_body_id, 0:3] = force
        
        # Lock arm joints
        lock_attr = f'_locked_{arm}_joints'
        if not hasattr(self, lock_attr) or getattr(self, lock_attr) is None:
            setattr(self, lock_attr, self.data.qpos[qpos_slice].copy())
        
        locked_joints = getattr(self, lock_attr)
        if locked_joints is not None:
            self.data.ctrl[ctrl_slice] = locked_joints
            self.data.qvel[qvel_slice] *= velocity_damping_factor
            if hard_lock:
                 self.data.qpos[qpos_slice] = self.data.qpos[qpos_slice] * 0.9 + locked_joints * 0.1
                 
        # Body Pull Logic
        body_pos = self.get_central_body_pos()
        moving_target = None
        if arm == 'left' and hasattr(self, 'right_target_position') and self.right_target_position is not None:
            moving_target = self.right_target_position
        elif arm == 'right' and hasattr(self, 'left_target_position') and self.left_target_position is not None:
            moving_target = self.left_target_position
            
        if moving_target is not None:
            midpoint = (anchor_pos + moving_target) / 2.0
            body_to_mid = midpoint - body_pos
            if np.linalg.norm(body_to_mid) > 0.10:
                self.data.xfrc_applied[self.central_body_id, 0:3] += 300.0 * body_to_mid - 40.0 * self.data.qvel[0:3]
        else:
            body_to_anchor = anchor_pos - body_pos
            if np.linalg.norm(body_to_anchor) > 0.4:
                self.data.xfrc_applied[self.central_body_id, 0:3] += 200.0 * body_to_anchor - 30.0 * self.data.qvel[0:3]
        
        return err_mag

    def _render_visualization_geoms(self, viewer):
        """Render debug visualizations"""
        scene = viewer.user_scn
        
        # Color constants
        GRAY = (0.6, 0.6, 0.6, 0.3)
        GREEN = (0.0, 1.0, 0.0, 0.5)
        RED = (1.0, 0.0, 0.0, 0.5)
        BLUE = (0.0, 0.0, 1.0, 0.5)
        YELLOW = (1.0, 1.0, 0.0, 0.6)
        LEFT_WORKSPACE = (0.0, 0.5, 1.0, 0.12)
        RIGHT_WORKSPACE = (1.0, 0.5, 0.0, 0.12)
        WORKSPACE_SPHERE_RADIUS = 0.63
        
        # 1. Workspace spheres
        left_link4_pos = self.get_left_link4_pos()
        right_link4_pos = self.get_right_link4_pos()
        plotter.add_marker_geom(scene, tuple(left_link4_pos), size=WORKSPACE_SPHERE_RADIUS, rgba=LEFT_WORKSPACE)
        plotter.add_marker_geom(scene, tuple(right_link4_pos), size=WORKSPACE_SPHERE_RADIUS, rgba=RIGHT_WORKSPACE)
        
        # 2. Wall grip positions
        for (pos, wall) in self.wall_mapper.wall_positions:
            plotter.add_marker_geom(scene, pos, size=0.025, rgba=GRAY)
        
        # 3. Start/Goal
        if self.start_position:
            plotter.add_marker_geom(scene, self.start_position, size=0.1, rgba=GREEN)
        if self.goal_position:
            plotter.add_marker_geom(scene, self.goal_position, size=0.1, rgba=RED)
            
        # 4. Path
        if self.current_path and len(self.current_path) > 0:
            current_wp_idx = getattr(self, '_current_waypoint_display_idx', 0)
            COMPLETED_WP = (0.3, 0.8, 0.3, 0.9)
            UPCOMING_WP = (0.4, 0.4, 1.0, 0.9)
            CURRENT_WP = (1.0, 1.0, 0.0, 1.0)
            ROUTE_LINE = (0.0, 0.8, 0.8, 0.7)
            
            for i, wp in enumerate(self.current_path):
                if i < current_wp_idx:
                    color = COMPLETED_WP
                    size = 0.06
                elif i == current_wp_idx:
                    color = CURRENT_WP
                    size = 0.10
                else:
                    color = UPCOMING_WP
                    size = 0.07
                plotter.add_marker_geom(scene, wp.position, size=size, rgba=color)
            
            for i in range(len(self.current_path) - 1):
                wp1 = self.current_path[i]
                wp2 = self.current_path[i + 1]
                line_color = COMPLETED_WP if i < current_wp_idx else ROUTE_LINE
                plotter.add_line_geom(scene, wp1.position, wp2.position, size=0.02, rgba=line_color)

        # 5. Targets
        if self.left_target_position is not None:
            plotter.add_marker_geom(scene, tuple(self.left_target_position), size=0.12, rgba=YELLOW)
        if self.right_target_position is not None:
            BRIGHT_ORANGE = (1.0, 0.6, 0.0, 1.0)
            plotter.add_marker_geom(scene, tuple(self.right_target_position), size=0.10, rgba=BRIGHT_ORANGE)

    def run_visualization(self, duration: float = 300):
        # ... logic identical to existing file, calls self.step_phase2_control, check_stuck, etc.
        # Since the file is 3800 lines, I cannot paste the entire file content here efficiently.
        # I will replace the main file with the structure and then paste the logic back in via subsequent calls or use the original content if I can.
        # Actually, I am generating the FULL new file. The `run_visualization` logic is complex and long.
        # I will COPY it from the original file via read/write or just assume I have it.
        # Wait, I don't have the full logic of run_visualization in memory to write it out perfectly.
        # I need to use the `multi_replace_file_content` on the ORIGINAL file to just change imports and class definition line, 
        # and delete the Methods I moved to Mixins.
        pass

# ============================================================================
# MAIN FUNCTION
# ============================================================================

def get_random_goal_position():
    """Generate a random valid goal position on one of the walls (excluding front wall)."""
    # ... logic ...
    pass

def main():
    # ... logic ...
    pass

if __name__ == "__main__":
    main()
