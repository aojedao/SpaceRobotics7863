"""
Arm Controller Module for Wall Crawler Robot

This module contains all arm control, IK computation, anchor management,
and coordinated arm control for the dual-arm wall-crawling robot.

Classes:
    - ArmControllerConfig: Configuration dataclass for controller gains
    - ArmController: Main controller class for dual-arm IK and management
"""

import numpy as np
import mujoco
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class ArmControllerConfig:
    """Configuration for arm controller gains and parameters"""
    # Position control gains
    kp_position: float = 400.0
    kd_position: float = 18.0
    kp_orientation: float = 30.0
    kd_orientation: float = 15.0
    lambda_dls: float = 0.012  # Damping for damped least squares
    
    # Per-joint gain scaling [joint1, joint2, joint3, joint4, joint5, joint6, joint7]
    joint_gain_scale: np.ndarray = None
    
    # Anchor control gains
    kp_anchor: float = 800.0
    kd_anchor: float = 40.0
    
    # Gripper values
    gripper_open_value: int = 0
    gripper_closed_value: int = 255
    
    def __post_init__(self):
        if self.joint_gain_scale is None:
            self.joint_gain_scale = np.array([2.0, 1.8, 1.2, 1.0, 1.0, 0.8, 0.6])


class ArmController:
    """
    Controller for dual-arm wall-crawling robot.
    
    Handles:
    - Inverse kinematics using Jacobian pseudoinverse
    - Gripper open/close control
    - Anchor management for wall attachment
    - Coordinated arm control for body repositioning
    - Arm crossing detection and prevention
    """
    
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, 
                 config: Optional[ArmControllerConfig] = None):
        """
        Initialize the arm controller.
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            config: Controller configuration (uses defaults if None)
        """
        self.model = model
        self.data = data
        self.config = config or ArmControllerConfig()
        
        # Setup body and actuator IDs
        self._setup_ids()
        
        # Target positions
        self.left_target_position: Optional[np.ndarray] = None
        self.right_target_position: Optional[np.ndarray] = None
        
        # Anchor state
        self.left_arm_anchored = False
        self.right_arm_anchored = False
        self.left_anchor_position: Optional[np.ndarray] = None
        self.right_anchor_position: Optional[np.ndarray] = None
        
        # Locked positions for anchor maintenance
        self.locked_body_pos: Optional[np.ndarray] = None
        self.locked_body_quat: Optional[np.ndarray] = None
        self.locked_left_arm_joints: Optional[np.ndarray] = None
        
        # Gripper state
        from enum import Enum, auto
        class GripperState(Enum):
            OPEN = auto()
            CLOSED = auto()
            MOVING = auto()
        
        self.GripperState = GripperState
        self.left_gripper_state = GripperState.CLOSED
        self.right_gripper_state = GripperState.CLOSED
        
        # Diagnostic counters
        self._crossing_diag_counter = 0
        self._cross_print_counter = 0
    
    def _setup_ids(self):
        """Setup body and actuator IDs from model"""
        # End-effector body IDs
        self.left_ee_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "left_gripper_base_mount"
        )
        self.right_ee_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "right_gripper_base_mount"
        )
        
        # Gripper site IDs
        self.left_gripper_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "left_gripper_center"
        )
        self.right_gripper_site_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SITE, "right_gripper_center"
        )
        
        # Central body ID
        self.central_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "central_body"
        )
        
        # Actuator slices
        self.left_arm_actuator_slice = slice(0, 7)
        self.right_arm_actuator_slice = slice(7, 14)
        self.left_gripper_actuator_idx = 14
        self.right_gripper_actuator_idx = 15
        
        # Joint position/velocity slices
        self.left_arm_qpos_slice = slice(7, 14)
        self.right_arm_qpos_slice = slice(22, 29)
        self.left_arm_qvel_slice = slice(6, 13)
        self.right_arm_qvel_slice = slice(21, 28)
    
    # =========================================================================
    # POSITION GETTERS
    # =========================================================================
    
    def get_left_gripper_pos(self) -> np.ndarray:
        """Get left gripper center position"""
        return self.data.site_xpos[self.left_gripper_site_id].copy()
    
    def get_right_gripper_pos(self) -> np.ndarray:
        """Get right gripper center position"""
        return self.data.site_xpos[self.right_gripper_site_id].copy()
    
    def get_central_body_pos(self) -> np.ndarray:
        """Get central body position"""
        return self.data.xpos[self.central_body_id].copy()
    
    def get_left_ee_orientation(self) -> np.ndarray:
        """Get left end-effector orientation as 3x3 rotation matrix"""
        return self.data.site_xmat[self.left_gripper_site_id].reshape(3, 3).copy()
    
    def get_right_ee_orientation(self) -> np.ndarray:
        """Get right end-effector orientation as 3x3 rotation matrix"""
        return self.data.site_xmat[self.right_gripper_site_id].reshape(3, 3).copy()
    
    # =========================================================================
    # SURFACE ORIENTATION
    # =========================================================================
    
    def get_surface_normal(self, wall: str) -> np.ndarray:
        """Get the inward-pointing surface normal for a wall.
        
        Args:
            wall: 'floor', 'ceiling', 'front', 'back', 'left_wall', 'right_wall'
            
        Returns:
            3D unit vector pointing INTO the module from the wall
        """
        normals = {
            'floor': np.array([0.0, 0.0, 1.0]),
            'ceiling': np.array([0.0, 0.0, -1.0]),
            'front': np.array([0.0, -1.0, 0.0]),
            'back': np.array([0.0, 1.0, 0.0]),
            'left_wall': np.array([1.0, 0.0, 0.0]),
            'right_wall': np.array([-1.0, 0.0, 0.0]),
        }
        return normals.get(wall, np.array([0.0, 0.0, -1.0]))
    
    def get_target_orientation_matrix(self, wall: str) -> np.ndarray:
        """Get desired end-effector orientation matrix for approaching a wall.
        
        Args:
            wall: Wall type string
            
        Returns:
            3x3 rotation matrix for desired gripper orientation
        """
        z_axis = -self.get_surface_normal(wall)
        
        if abs(z_axis[2]) < 0.9:
            up = np.array([0.0, 0.0, 1.0])
        else:
            up = np.array([0.0, 1.0, 0.0])
        
        x_axis = np.cross(up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        return np.column_stack([x_axis, y_axis, z_axis])
    
    # =========================================================================
    # JACOBIAN AND IK COMPUTATION
    # =========================================================================
    
    def compute_jacobian(self, body_id: int) -> Tuple[np.ndarray, np.ndarray]:
        """Compute the Jacobian for a given body"""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, body_id)
        return jacp, jacr
    
    def compute_first_joint_target(self, arm: str, target_pos: np.ndarray, 
                                    relaxed_limits: bool = False) -> float:
        """Compute the ideal first joint angle to orient the arm toward target.
        
        Args:
            arm: 'left' or 'right'
            target_pos: Target position in world coordinates
            relaxed_limits: If True, allow wider range for coordinated control
            
        Returns:
            Ideal first joint angle in radians
        """
        body_pos = self.get_central_body_pos()
        body_quat = self.data.qpos[3:7]
        
        # Arms mounted on Y-axis sides (front/back of central body)
        if arm == 'left':
            arm_base_offset = np.array([0, -0.15, 0])  # Left arm on front side (-Y)
        else:
            arm_base_offset = np.array([0, 0.15, 0])   # Right arm on back side (+Y)
        
        arm_base = body_pos + arm_base_offset
        to_target = target_pos - arm_base
        to_target_xy = np.array([to_target[0], to_target[1]])
        
        if np.linalg.norm(to_target_xy) < 0.1:
            return 0.0
        
        world_angle = np.arctan2(to_target_xy[1], to_target_xy[0])
        
        w, x, y, z = body_quat
        body_yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        
        joint1_angle = world_angle - body_yaw
        
        while joint1_angle > np.pi:
            joint1_angle -= 2*np.pi
        while joint1_angle < -np.pi:
            joint1_angle += 2*np.pi
        
        if relaxed_limits:
            max_limit = np.pi * 0.6
        else:
            max_limit = np.pi * 0.4
        
        if arm == 'left':
            if joint1_angle > max_limit:
                joint1_angle = max_limit
        else:
            if joint1_angle < -max_limit:
                joint1_angle = -max_limit
        
        return joint1_angle
    
    def detect_arm_crossing(self, verbose: bool = False) -> Tuple[bool, float]:
        """Detect if the arms are crossing each other.
        
        Returns:
            Tuple of (is_crossing, crossing_severity)
        """
        body_pos = self.get_central_body_pos()
        body_quat = self.data.qpos[3:7]
        
        left_grip = self.get_left_gripper_pos()
        right_grip = self.get_right_gripper_pos()
        
        w, x, y, z = body_quat
        body_yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        
        cos_yaw = np.cos(-body_yaw)
        sin_yaw = np.sin(-body_yaw)
        
        left_rel = left_grip - body_pos
        right_rel = right_grip - body_pos
        
        left_local_x = left_rel[0] * cos_yaw - left_rel[1] * sin_yaw
        right_local_x = right_rel[0] * cos_yaw - right_rel[1] * sin_yaw
        left_local_y = left_rel[0] * sin_yaw + left_rel[1] * cos_yaw
        right_local_y = right_rel[0] * sin_yaw + right_rel[1] * cos_yaw
        
        left_on_wrong_side = left_local_x > 0.2
        right_on_wrong_side = right_local_x < -0.2
        
        gripper_distance = np.linalg.norm(left_grip - right_grip)
        grippers_too_close = gripper_distance < 0.15
        
        is_crossing = (left_on_wrong_side and right_on_wrong_side) or grippers_too_close
        
        severity = 0.0
        if left_on_wrong_side and right_on_wrong_side:
            severity += min((left_local_x - 0.1) / 0.4, 0.5)
            severity += min((-right_local_x - 0.1) / 0.4, 0.5)
        if grippers_too_close:
            severity += (0.15 - gripper_distance) / 0.15
        
        severity = min(severity, 1.0)
        
        if is_crossing and verbose:
            self._crossing_diag_counter += 1
            if self._crossing_diag_counter % 200 == 1:
                print(f"\n  ╔══════════════════════════════════════════════════════════╗")
                print(f"  ║           ⚠️  ARM CROSSING DIAGNOSTICS                    ║")
                print(f"  ╠══════════════════════════════════════════════════════════╣")
                print(f"  ║ Body Position: ({body_pos[0]:.3f}, {body_pos[1]:.3f}, {body_pos[2]:.3f})")
                print(f"  ║ Body Yaw: {np.degrees(body_yaw):.1f}°")
                print(f"  ╟──────────────────────────────────────────────────────────╢")
                print(f"  ║ LEFT Gripper (world):  ({left_grip[0]:.3f}, {left_grip[1]:.3f}, {left_grip[2]:.3f})")
                print(f"  ║ LEFT Gripper (local):  X={left_local_x:+.3f}, Y={left_local_y:+.3f}")
                print(f"  ║   → Should be X < -0.2 | {'❌ ON WRONG SIDE' if left_on_wrong_side else '✓ OK'}")
                print(f"  ╟──────────────────────────────────────────────────────────╢")
                print(f"  ║ RIGHT Gripper (world): ({right_grip[0]:.3f}, {right_grip[1]:.3f}, {right_grip[2]:.3f})")
                print(f"  ║ RIGHT Gripper (local): X={right_local_x:+.3f}, Y={right_local_y:+.3f}")
                print(f"  ║   → Should be X > +0.2 | {'❌ ON WRONG SIDE' if right_on_wrong_side else '✓ OK'}")
                print(f"  ╟──────────────────────────────────────────────────────────╢")
                print(f"  ║ Gripper Distance: {gripper_distance:.3f}m {'⚠️ TOO CLOSE' if grippers_too_close else ''}")
                print(f"  ║ Crossing Severity: {severity:.2f} / 1.00")
                reason = []
                if left_on_wrong_side and right_on_wrong_side:
                    reason.append("BOTH ARMS ON WRONG SIDES")
                if grippers_too_close:
                    reason.append("GRIPPERS TOO CLOSE")
                print(f"  ║ Reason: {' + '.join(reason)}")
                print(f"  ╚══════════════════════════════════════════════════════════╝\n")
        elif not is_crossing:
            self._crossing_diag_counter = 0
        
        return is_crossing, severity
    
    def compute_arm_control(self, arm: str = 'left') -> Tuple[np.ndarray, np.ndarray]:
        """Compute Cartesian space control for specified arm.
        
        Returns:
            (joint_position_command, position_error)
        """
        if arm == 'left':
            current_pos = self.get_left_gripper_pos()
            target_pos = self.left_target_position
            body_id = self.left_ee_body_id
            qpos_slice = self.left_arm_qpos_slice
            qvel_slice = self.left_arm_qvel_slice
        else:
            current_pos = self.get_right_gripper_pos()
            target_pos = self.right_target_position
            body_id = self.right_ee_body_id
            qpos_slice = self.right_arm_qpos_slice
            qvel_slice = self.right_arm_qvel_slice
        
        if target_pos is None:
            return self.data.qpos[qpos_slice].copy(), np.zeros(3)
        
        pos_error = target_pos - current_pos
        
        jacp, jacr = self.compute_jacobian(body_id)
        
        if arm == 'left':
            arm_jacp = jacp[:, 6:13]
        else:
            arm_jacp = jacp[:, 21:28]
        
        J = arm_jacp
        task_error = self.config.kp_position * pos_error
        current_joint_vel = self.data.qvel[qvel_slice]
        
        JJT = J @ J.T
        J_pinv = J.T @ np.linalg.inv(JJT + self.config.lambda_dls**2 * np.eye(3))
        
        joint_vel_cmd = J_pinv @ task_error
        joint_vel_cmd = joint_vel_cmd * self.config.joint_gain_scale
        joint_vel_cmd -= self.config.kd_position * current_joint_vel
        
        current_joint_pos = self.data.qpos[qpos_slice]
        joint_pos_cmd = current_joint_pos + joint_vel_cmd * self.model.opt.timestep
        
        # First joint prioritization
        arm_is_anchored = (arm == 'left' and self.left_arm_anchored) or \
                          (arm == 'right' and self.right_arm_anchored)
        
        is_crossing, crossing_severity = self.detect_arm_crossing(verbose=True)
        
        ideal_j1 = self.compute_first_joint_target(arm, target_pos)
        current_j1 = current_joint_pos[0]
        j1_error = ideal_j1 - current_j1
        
        while j1_error > np.pi:
            j1_error -= 2*np.pi
        while j1_error < -np.pi:
            j1_error += 2*np.pi
        
        if not arm_is_anchored:
            if arm == 'left':
                if current_j1 > 0.3 and j1_error > 0:
                    j1_error = -0.5 - current_j1
                    j1_correction_gain = 20.0
                    if is_crossing:
                        self._cross_print_counter += 1
                        if self._cross_print_counter % 100 == 1:
                            print(f"  ⚠️ LEFT arm on wrong side! Pushing back to left")
                else:
                    pos_error_mag = np.linalg.norm(pos_error)
                    j1_correction_gain = 12.0 if pos_error_mag > 0.3 else 4.0
            else:
                if current_j1 < -0.3 and j1_error < 0:
                    j1_error = 0.5 - current_j1
                    j1_correction_gain = 20.0
                    if is_crossing:
                        self._cross_print_counter += 1
                        if self._cross_print_counter % 100 == 1:
                            print(f"  ⚠️ RIGHT arm on wrong side! Pushing back to right")
                else:
                    pos_error_mag = np.linalg.norm(pos_error)
                    j1_correction_gain = 12.0 if pos_error_mag > 0.3 else 4.0
        else:
            j1_correction_gain = 4.0
        
        dt = self.model.opt.timestep
        delta_j1 = j1_correction_gain * j1_error * dt
        max_delta = 0.10
        delta_j1 = np.clip(delta_j1, -max_delta, max_delta)
        joint_pos_cmd[0] += delta_j1
        
        return joint_pos_cmd, pos_error
    
    def apply_arm_control(self, arm: str = 'left', skip_base_joints: int = 0, 
                          anchored_mode: bool = False) -> float:
        """Apply arm control to reach target position.
        
        Args:
            arm: 'left' or 'right'
            skip_base_joints: Number of base joints to skip
            anchored_mode: If True, only control last 2 joints
            
        Returns:
            Position error magnitude
        """
        joint_cmd, pos_error = self.compute_arm_control(arm)
        
        if anchored_mode:
            skip_base_joints = 5
        
        if arm == 'left':
            if skip_base_joints > 0:
                current_pos = self.data.qpos[7:7+skip_base_joints]
                joint_cmd[:skip_base_joints] = current_pos
            self.data.ctrl[self.left_arm_actuator_slice] = joint_cmd
        else:
            if skip_base_joints > 0:
                current_pos = self.data.qpos[22:22+skip_base_joints]
                joint_cmd[:skip_base_joints] = current_pos
            self.data.ctrl[self.right_arm_actuator_slice] = joint_cmd
        
        return np.linalg.norm(pos_error)
    
    def apply_arm_control_with_orientation(self, arm: str, target_wall: str, 
                                           skip_base_joints: int = 0) -> float:
        """Apply arm control with orientation alignment.
        
        Args:
            arm: 'left' or 'right'
            target_wall: Wall type for orientation alignment
            skip_base_joints: Number of base joints to skip
            
        Returns:
            Position error magnitude
        """
        if arm == 'left':
            current_pos = self.get_left_gripper_pos()
            current_orient = self.get_left_ee_orientation()
            target_pos = self.left_target_position
            body_id = self.left_ee_body_id
            qpos_slice = self.left_arm_qpos_slice
            qvel_slice = self.left_arm_qvel_slice
            actuator_slice = self.left_arm_actuator_slice
        else:
            current_pos = self.get_right_gripper_pos()
            current_orient = self.get_right_ee_orientation()
            target_pos = self.right_target_position
            body_id = self.right_ee_body_id
            qpos_slice = self.right_arm_qpos_slice
            qvel_slice = self.right_arm_qvel_slice
            actuator_slice = self.right_arm_actuator_slice
        
        if target_pos is None:
            return 0.0
        
        pos_error = target_pos - current_pos
        target_orient = self.get_target_orientation_matrix(target_wall)
        orient_error_matrix = target_orient @ current_orient.T
        
        trace = np.trace(orient_error_matrix)
        angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
        
        if angle > 1e-6:
            axis = np.array([
                orient_error_matrix[2, 1] - orient_error_matrix[1, 2],
                orient_error_matrix[0, 2] - orient_error_matrix[2, 0],
                orient_error_matrix[1, 0] - orient_error_matrix[0, 1]
            ])
            axis = axis / (2 * np.sin(angle) + 1e-10)
            orient_error = angle * axis
        else:
            orient_error = np.zeros(3)
        
        jacp, jacr = self.compute_jacobian(body_id)
        
        if arm == 'left':
            arm_jacp = jacp[:, 6:13]
            arm_jacr = jacr[:, 6:13]
        else:
            arm_jacp = jacp[:, 21:28]
            arm_jacr = jacr[:, 21:28]
        
        J = np.vstack([arm_jacp, arm_jacr])
        task_error = np.concatenate([
            self.config.kp_position * pos_error,
            self.config.kp_orientation * orient_error
        ])
        
        current_joint_vel = self.data.qvel[qvel_slice]
        
        JJT = J @ J.T
        J_pinv = J.T @ np.linalg.inv(JJT + self.config.lambda_dls**2 * np.eye(6))
        
        joint_vel_cmd = J_pinv @ task_error
        joint_vel_cmd -= self.config.kd_position * current_joint_vel
        
        current_joint_pos = self.data.qpos[qpos_slice]
        joint_pos_cmd = current_joint_pos + joint_vel_cmd * self.model.opt.timestep
        
        if skip_base_joints > 0:
            if arm == 'left':
                current_base = self.data.qpos[7:7+skip_base_joints]
            else:
                current_base = self.data.qpos[22:22+skip_base_joints]
            joint_pos_cmd[:skip_base_joints] = current_base
        
        self.data.ctrl[actuator_slice] = joint_pos_cmd
        
        return np.linalg.norm(pos_error)
    
    def apply_coordinated_arm_control(self, anchored_arm: str, moving_target: np.ndarray, 
                                      anchor_pos: np.ndarray):
        """Use first 5 joints of anchored arm to help body reach moving arm's target.
        
        Args:
            anchored_arm: 'left' or 'right' - the arm that is anchored
            moving_target: The target position the moving arm is trying to reach
            anchor_pos: The anchor position that must be maintained
        """
        body_pos = self.get_central_body_pos()
        body_to_target = moving_target - body_pos
        target_distance = np.linalg.norm(body_to_target)
        
        if target_distance < 0.2:
            return
        
        if anchored_arm == 'left':
            body_id = self.left_ee_body_id
            qpos_slice = self.left_arm_qpos_slice
            qvel_slice = self.left_arm_qvel_slice
            actuator_slice = self.left_arm_actuator_slice
        else:
            body_id = self.right_ee_body_id
            qpos_slice = self.right_arm_qpos_slice
            qvel_slice = self.right_arm_qvel_slice
            actuator_slice = self.right_arm_actuator_slice
        
        body_direction = body_to_target / target_distance
        desired_body_vel = body_direction * min(target_distance, 0.8)
        
        jacp_full, _ = self.compute_jacobian(body_id)
        
        if anchored_arm == 'left':
            J_base = jacp_full[:, 6:11]
        else:
            J_base = jacp_full[:, 21:26]
        
        kp_coord = 2500.0
        kd_coord = 1.5
        
        task_vel = -kp_coord * desired_body_vel
        
        lambda_coord = 0.08
        JJT = J_base @ J_base.T
        J_pinv = J_base.T @ np.linalg.inv(JJT + lambda_coord**2 * np.eye(3))
        
        joint_vel_base = J_pinv @ task_vel
        
        current_joint_vel = self.data.qvel[qvel_slice][:5]
        joint_vel_base -= kd_coord * current_joint_vel
        
        current_q = self.data.qpos[qpos_slice]
        dt = self.model.opt.timestep
        
        for i in range(5):
            new_q = current_q[i] + joint_vel_base[i] * dt
            self.data.ctrl[actuator_slice.start + i] = new_q
    
    # =========================================================================
    # GRIPPER CONTROL
    # =========================================================================
    
    def set_gripper(self, arm: str, close: bool):
        """Set gripper state (open or closed)
        
        Args:
            arm: 'left' or 'right'
            close: True to close gripper, False to open
        """
        value = self.config.gripper_closed_value if close else self.config.gripper_open_value
        
        if arm == 'left':
            self.data.ctrl[self.left_gripper_actuator_idx] = value
            self.left_gripper_state = self.GripperState.CLOSED if close else self.GripperState.OPEN
            if not close:
                self.release_anchor('left')
        else:
            self.data.ctrl[self.right_gripper_actuator_idx] = value
            self.right_gripper_state = self.GripperState.CLOSED if close else self.GripperState.OPEN
            if not close:
                self.release_anchor('right')
        
        state_str = "CLOSED" if close else "OPEN"
        print(f"  🤏 {arm.upper()} gripper: {state_str}")
    
    def get_gripper_closure(self, arm: str) -> float:
        """Get how closed the gripper is (0.0 = open, 1.0 = closed)"""
        if arm == 'left':
            ctrl_value = self.data.ctrl[self.left_gripper_actuator_idx]
        else:
            ctrl_value = self.data.ctrl[self.right_gripper_actuator_idx]
        
        closure = ctrl_value / self.config.gripper_closed_value
        return np.clip(closure, 0.0, 1.0)
    
    def is_gripper_closed_enough(self, arm: str, threshold: float = 0.8) -> bool:
        """Check if gripper is closed enough to maintain anchor"""
        return self.get_gripper_closure(arm) >= threshold
    
    # =========================================================================
    # ANCHOR MANAGEMENT
    # =========================================================================
    
    def anchor_gripper(self, arm: str, position: np.ndarray):
        """Anchor the gripper at a specific world position"""
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
        """Release the anchor on specified arm"""
        if arm == 'left':
            if self.left_arm_anchored:
                print(f"  🔓 {arm.upper()} arm RELEASED from anchor")
            self.left_arm_anchored = False
            self.left_anchor_position = None
            if hasattr(self, '_locked_left_joints'):
                self._locked_left_joints = None
        else:
            if self.right_arm_anchored:
                print(f"  🔓 {arm.upper()} arm RELEASED from anchor")
            self.right_arm_anchored = False
            self.right_anchor_position = None
            if hasattr(self, '_locked_right_joints'):
                self._locked_right_joints = None
    
    def maintain_anchor(self, arm: str, control_anchored_arm: bool = False) -> float:
        """Maintain the anchor by locking body position and arm joints.
        
        Returns:
            Error magnitude from anchor position
        """
        if arm == 'left' and self.left_arm_anchored:
            if not self.is_gripper_closed_enough('left'):
                self.release_anchor('left')
                return 0.0
            
            if self.locked_body_pos is not None:
                self.data.qpos[0:3] = self.locked_body_pos.copy()
                self.data.qpos[3:7] = self.locked_body_quat.copy()
                self.data.qvel[0:6] = 0.0
                
                if self.locked_left_arm_joints is not None:
                    self.data.qpos[7:14] = self.locked_left_arm_joints.copy()
                    self.data.qvel[6:13] = 0.0
                
                mujoco.mj_forward(self.model, self.data)
            
            anchor_pos = self.left_anchor_position
            gripper_pos = self.get_left_gripper_pos()
            error_magnitude = np.linalg.norm(anchor_pos - gripper_pos)
            
            if control_anchored_arm:
                self.apply_arm_control('left')
            
            return error_magnitude
            
        elif arm == 'right' and self.right_arm_anchored:
            if not self.is_gripper_closed_enough('right'):
                self.release_anchor('right')
                return 0.0
            
            if self.locked_body_pos is not None:
                self.data.qpos[0:3] = self.locked_body_pos.copy()
                self.data.qpos[3:7] = self.locked_body_quat.copy()
                self.data.qvel[0:6] = 0.0
                mujoco.mj_forward(self.model, self.data)
            
            anchor_pos = self.right_anchor_position
            gripper_pos = self.get_right_gripper_pos()
            error_magnitude = np.linalg.norm(anchor_pos - gripper_pos)
            
            if control_anchored_arm:
                self.apply_arm_control('right')
            
            return error_magnitude
        
        return 0.0
    
    def lock_body_position(self):
        """Lock the current body position and left arm joints for anchor constraint."""
        self.locked_body_pos = self.data.qpos[0:3].copy()
        self.locked_body_quat = self.data.qpos[3:7].copy()
        self.locked_left_arm_joints = self.data.qpos[7:14].copy()
        print(f"  🔒 Body locked at: ({self.locked_body_pos[0]:.2f}, {self.locked_body_pos[1]:.2f}, {self.locked_body_pos[2]:.2f})")
        left_grip = self.get_left_gripper_pos()
        print(f"  🔒 Left gripper at: ({left_grip[0]:.2f}, {left_grip[1]:.2f}, {left_grip[2]:.2f})")
    
    def step_phase2_control(self):
        """Step the Phase 2 controller - anchor arms and maintain position"""
        left_anchored = self.left_arm_anchored and self.is_gripper_closed_enough('left')
        right_anchored = self.right_arm_anchored and self.is_gripper_closed_enough('right')
        
        if left_anchored and right_anchored:
            self.data.qvel[0:6] = 0.0
            self.apply_arm_control('left')
            self.apply_arm_control('right')
        elif left_anchored:
            self.maintain_anchor('left', control_anchored_arm=False)
        elif right_anchored:
            self.maintain_anchor('right', control_anchored_arm=False)
