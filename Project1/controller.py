#!/usr/bin/env python3
"""
Robot Controller Module for Integrated Zero-Gravity Simulation

This module provides various controller implementations for the KUKA iiwa14 robot,
including position control and advanced torque-balancing control.
"""

import mujoco
import numpy as np
import time
from scipy.spatial.transform import Rotation
from collections import deque
from abc import ABC, abstractmethod


class BaseController(ABC):
    """Abstract base class for robot controllers"""
    
    def __init__(self, model, data):
        """
        Initialize the base controller
        
        Args:
            model: MuJoCo model object
            data: MuJoCo data object
        """
        self.model = model
        self.data = data
        self.arm_joint_names = [
            "joint1", "joint2", "joint3", "joint4",
            "joint5", "joint6", "joint7"
        ]
        self.arm_joint_ids = self._setup_joint_ids()
        self.enabled = True
        self.previous_position_error = np.zeros(3)
        self.gripper_target = 0.0
        # Detect if a manual actuator for joint7 exists (added to XML as 'manual_joint7')
        try:
            self.manual_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "manual_joint7")
            if self.manual_actuator_id < 0:
                self.manual_actuator_id = None
            else:
                print(f"✓ Manual actuator 'manual_joint7' detected at actuator ID {self.manual_actuator_id}")
        except Exception as e:
            self.manual_actuator_id = None
            print(f"⚠️ Could not detect manual_joint7 actuator: {e}")
        
    def _setup_joint_ids(self):
        """Setup joint IDs from joint names"""
        arm_joint_ids = []
        for name in self.arm_joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            arm_joint_ids.append(joint_id)
        return arm_joint_ids
    
    def get_end_effector_position(self):
        """Get current end-effector position"""
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripper_center")
        return self.data.site_xpos[site_id]
    
    def get_end_effector_orientation(self):
        """Get current end-effector orientation as rotation matrix"""
        orient_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripper_center")
        return self.data.site_xmat[orient_id].reshape(3, 3)
    
    def get_target_position(self):
        """Get target position from door handle site"""
        handle_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'door_handle_site')
        if handle_site_id >= 0:
            return self.data.site_xpos[handle_site_id]
        return np.array([0.3, 0.5, 0.5])
    
    def get_target_orientation(self):
        """Get target orientation from door handle site"""
        box_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "door_handle_site")
        if box_id >= 0:
            target_quat = np.array([-1, 0, -1, 0]) * self.data.xquat[box_id].copy()
            if np.linalg.norm(target_quat) > 1e-6:
                target_orient_mat = Rotation.from_quat(target_quat[[1, 2, 3, 0]]).as_matrix()
            else:
                target_orient_mat = np.eye(3)
        else:
            target_orient_mat = np.eye(3)
        return target_orient_mat
    
    def compute_jacobian(self):
        """Compute the Jacobian matrix for the robot arm"""
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripper_center")
        
        if site_id >= 0:
            jac_pos = np.zeros((3, self.model.nv))
            jac_rot = np.zeros((3, self.model.nv))
            mujoco.mj_jacSite(self.model, self.data, jac_pos, jac_rot, site_id)
            jacobian = np.vstack([jac_pos, jac_rot])
            return jacobian[:, :7]
        else:
            return np.eye(6, 7)
    
    def compute_position_error(self, current_pos, target_pos, approach_offset=None):
        """
        Compute position error for tracking.
        
        Args:
            current_pos: Current end-effector position
            target_pos: Target position (e.g., box/door handle)
            approach_offset: Optional offset for approach position (default: None, track directly)
            
        Returns:
            position_error: 3D position error vector
        """
        if approach_offset is not None:
            return target_pos - current_pos + approach_offset
        return target_pos - current_pos
    
    def compute_orientation_error(self, current_orient_mat, target_orient_mat):
        """
        Compute orientation error between current and target orientations.
        
        Args:
            current_orient_mat: Current orientation as 3x3 rotation matrix
            target_orient_mat: Target orientation as 3x3 rotation matrix
            
        Returns:
            orientation_error: 3D orientation error vector (axis-angle representation)
        """
        R_error = current_orient_mat.T @ target_orient_mat
        orientation_error = np.array([
            R_error[2, 1] - R_error[1, 2],
            R_error[0, 2] - R_error[2, 0],
            R_error[1, 0] - R_error[0, 1]
        ]) * 0.5
        return orientation_error
    
    def apply_control_vector(self, command_vector):
        """
        Apply a control vector to self.data.ctrl while respecting manual actuator.
        
        If a manual actuator (named 'manual_joint7') exists, its ctrl entry will not
        be overwritten so the viewer slider can be used to inject manual torque.
        
        Args:
            command_vector: Array of control commands for arm joints
        """
        n_ctrl = len(self.data.ctrl)
        manual_id = getattr(self, 'manual_actuator_id', None)
        
        # Apply command vector (skip manual actuator if it exists)
        for i, val in enumerate(command_vector):
            if i < n_ctrl:
                if manual_id is not None and i == manual_id:
                    continue
                self.data.ctrl[i] = float(val)
        
        # Zero remaining ctrl entries (except manual actuator)
        for idx in range(len(command_vector), n_ctrl):
            if manual_id is not None and idx == manual_id:
                continue
            self.data.ctrl[idx] = 0.0
    
    @abstractmethod
    def compute_control(self):
        """Compute control commands - must be implemented by subclasses"""
        pass


class PositionController(BaseController):
    """
    Standard position controller using Cartesian space control with quaternion orientation.
    
    Features:
    - 6DOF Cartesian space control (dynamically tracks target wherever it moves)
    - Quaternion-based orientation control
    - Redundant manipulator control with null space projection
    - Damped least squares to avoid singularities
    - Angular velocity damping for orientation stability
    """
    
    # Rotation correction matrix for aligning gripper frame to target frame
    ROTATION_CORRECTOR = np.array([[0, -1, 1], [0, 0, -1], [-1, 0, 0]])
    
    def __init__(self, model, data, **kwargs):
        """
        Initialize position controller
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            **kwargs: Additional parameters:
                - kp_position: Position gain (default: 100.0)
                - kd_position: Position derivative gain (default: 20.0)
                - max_joint_velocity: Maximum joint velocity (default: 2.0)
                - approach_offset: Optional 3D offset for approach position (default: None)
        """
        super().__init__(model, data)
        self.kp_position = kwargs.get('kp_position', 100.0)
        self.kd_position = kwargs.get('kd_position', 20.0)
        self.max_joint_velocity = kwargs.get('max_joint_velocity', 2.0)
        self.approach_offset = kwargs.get('approach_offset', None)  # Dynamic tracking by default
        self.target_position = None
        self.target_orientation = None
        self.current_target_orientation = None
        
        # Controller gains
        self.K_pos = np.diag([2.0, 2.5, 1.5]) * 5.0
        self.K_angular_vel = np.diag([1.0, 1.0, 1.0]) * 0.02
        self.K_orient_error = np.diag([1.5, 1.5, 0.0]) * 0.5
        self.lambda_damping = 0.25
    
    def compute_control(self):
        """Compute position control commands - dynamically tracks target position"""
        if not self.enabled:
            return
        
        # Get current state
        current_pos = self.get_end_effector_position()
        current_orient_mat = self.get_end_effector_orientation()
        self.target_position = self.get_target_position()
        self.target_orientation = self.get_target_orientation()
        current_joint_vel = self.data.qvel[:7]
        
        # Compute Jacobian and angular velocity
        J = self.compute_jacobian()
        J_rot = J[3:6, :7]
        current_angular_vel = J_rot @ current_joint_vel
        
        # Calculate position error (dynamically tracks target)
        position_error = self.compute_position_error(
            current_pos, self.target_position, self.approach_offset
        )
        
        # Apply rotation correction and compute orientation error
        target_orient_mat = self.ROTATION_CORRECTOR @ self.target_orientation
        orientation_error = self.compute_orientation_error(current_orient_mat, target_orient_mat)
        
        # Calculate angular velocity error
        target_angular_vel = self.data.qvel[3:6] if self.model.nv > 6 else np.zeros(3)
        angular_vel_err = current_angular_vel - target_angular_vel
        
        # Compute desired Cartesian velocities
        desired_position_velocity = self.K_pos @ position_error
        desired_angular_velocity = (self.K_orient_error @ orientation_error) - (self.K_angular_vel @ angular_vel_err)
        desired_cartesian_velocity = np.concatenate([desired_position_velocity, desired_angular_velocity])
        
        # Map to joint space with damped least squares
        try:
            J_damped_pinv = J.T @ np.linalg.inv(J @ J.T + self.lambda_damping * np.eye(6))
            joint_velocities = J_damped_pinv @ desired_cartesian_velocity
        except np.linalg.LinAlgError:
            joint_velocities = np.zeros(7)
        
        # Apply velocity limits and zero out joint 7 (reserved for manual control)
        joint_velocities_clipped = np.clip(joint_velocities, -self.max_joint_velocity, self.max_joint_velocity)
        joint_velocities_clipped[6] = 0.0
        
        # Apply control commands
        self.apply_control_vector(joint_velocities_clipped)
        
        # Store for visualization
        self.current_target_orientation = target_orient_mat
        self.previous_position_error = position_error.copy()
        
        return {
            'current_pos': current_pos,
            'current_orient': current_orient_mat,
            'target_orient': target_orient_mat,
            'position_error': position_error,
            'orientation_error': orientation_error,
            'current_joint_vel': current_joint_vel,
            'current_angular_vel': current_angular_vel
        }


class TorqueBalancingController(BaseController):
    """
    Advanced torque-balancing controller that minimizes base shear forces.
    
    Features:
    - Dynamically tracks target position (same as PositionController)
    - Torque calculation and balancing
    - Base shear force minimization using moment-based cost function
    - Null space optimization for force distribution
    """
    
    # Shared rotation correction matrix
    ROTATION_CORRECTOR = np.array([[0, -1, 1], [0, 0, -1], [-1, 0, 0]])
    
    def __init__(self, model, data, **kwargs):
        """
        Initialize torque-balancing controller
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            **kwargs: Additional parameters:
                - max_joint_velocity: Maximum joint velocity (default: 2.0)
                - approach_offset: Optional 3D offset for approach position (default: None)
                - torque_balance_gain: Gain for moment balancing (default: 0.5)
        """
        super().__init__(model, data)
        self.max_joint_velocity = kwargs.get('max_joint_velocity', 2.0)
        self.approach_offset = kwargs.get('approach_offset', None)
        self.torque_balance_gain = kwargs.get('torque_balance_gain', 0.5)
        self.target_position = None
        self.target_orientation = None
        self.current_target_orientation = None
        
        # Controller gains (different from PositionController for torque balancing)
        self.K_pos = np.diag([8.2, 10.2, 7.0]) * 0.5
        self.K_angular_vel = np.diag([1.0, 1.0, 1.0]) * 0.005
        self.K_orient_error = np.diag([5.0, 5.0, 0.0]) * 0.05
        self.lambda_damping = 0.25
        
    def compute_joint_torques(self):
        """
        Compute torques acting on each joint using inverse dynamics.
        
        Returns:
            torques: 7-element array of joint torques
        """
        torques = np.zeros(7)
        for i in range(7):
            if i < len(self.data.ctrl) and i < len(self.model.actuator_gear):
                gear_ratio = self.model.actuator_gear[i, 0] if len(self.model.actuator_gear[i]) > 0 else 1.0
                torques[i] = self.data.ctrl[i] * gear_ratio
        return torques
    
    def compute_projected_moment_to_base(self):
        """
        Compute the moment from the 6th actuator projected to the base.
        
        Returns:
            projected_moment: Projected moment magnitude at base
            j6_rot: Rotational Jacobian column for joint 6
            tau_6: Current torque in 6th axis
        """
        J = self.compute_jacobian()
        J_rot = J[3:6, :7]
        tau_6 = self.data.ctrl[5] if len(self.data.ctrl) > 5 else 0.0
        j6_rot = J_rot[:, 5]
        projected_moment = np.linalg.norm(j6_rot) * np.abs(tau_6)
        return projected_moment, j6_rot, tau_6
    
    def compute_base_moment_cost(self):
        """
        Compute potential field cost for base moment minimization.
        
        Cost function: 1/2 * (tau_7 - projected_moment_from_tau_6)^2
        
        Returns:
            cost: Potential field cost value
            moment_error: Difference between tau_7 and projected_moment
            cost_data: Dictionary with detailed cost information
        """
        tau_7 = self.data.ctrl[8] if len(self.data.ctrl) > 6 else 0.0
        projected_moment, j6_rot, tau_6 = self.compute_projected_moment_to_base()
        
        moment_error = tau_7 - projected_moment
        cost = 0.5 * moment_error**2
        
        return cost, moment_error, {
            'tau_7': tau_7,
            'tau_6': tau_6,
            'projected_moment': projected_moment,
            'moment_error': moment_error,
            'cost': cost,
            'cost_grad_tau7': moment_error,
            'cost_grad_tau6': -moment_error * np.linalg.norm(j6_rot),
            'j6_rot': j6_rot
        }
    
    def compute_null_space_torque_correction(self, joint_torques):
        """
        Compute null space torque corrections using gradient of moment cost function.
        
        Args:
            joint_torques: 7-element array of current joint torques
            
        Returns:
            correction_torques: Corrective torques in null space to minimize base moment
        """
        J = self.compute_jacobian()
        
        try:
            J_pinv = np.linalg.pinv(J)
            N = np.eye(7) - J_pinv @ J
        except np.linalg.LinAlgError:
            N = np.eye(7)
        
        cost, moment_error, cost_data = self.compute_base_moment_cost()
        j6_rot = cost_data['j6_rot']
        
        correction_vector = np.zeros(7)
        if np.abs(moment_error) > 1e-6:
            correction_gain = self.torque_balance_gain
            correction_vector[5] = -moment_error * correction_gain * np.linalg.norm(j6_rot)
            correction_vector[6] = -moment_error * correction_gain
        
        return N @ correction_vector
    
    def compute_control(self):
        """Compute torque-balancing control commands using moment minimization"""
        if not self.enabled:
            return
        
        # Get current state
        current_pos = self.get_end_effector_position()
        current_orient_mat = self.get_end_effector_orientation()
        self.target_position = self.get_target_position()
        self.target_orientation = self.get_target_orientation()
        current_joint_vel = self.data.qvel[:7]
        
        # Compute Jacobian and angular velocity
        J = self.compute_jacobian()
        J_rot = J[3:6, :7]
        current_angular_vel = J_rot @ current_joint_vel
        
        # Calculate errors using shared methods (dynamic tracking)
        position_error = self.compute_position_error(
            current_pos, self.target_position, self.approach_offset
        )
        target_orient_mat = self.ROTATION_CORRECTOR @ self.target_orientation
        orientation_error = self.compute_orientation_error(current_orient_mat, target_orient_mat)
        
        # Angular velocity error
        target_angular_vel = self.data.qvel[3:6] if self.model.nv > 6 else np.zeros(3)
        angular_vel_err = current_angular_vel - target_angular_vel
        
        # Compute desired Cartesian velocities
        desired_position_velocity = self.K_pos @ position_error
        desired_angular_velocity = (self.K_orient_error @ orientation_error) - (self.K_angular_vel @ angular_vel_err)
        desired_cartesian_velocity = np.concatenate([desired_position_velocity, desired_angular_velocity])
        
        # Map to joint space with damped least squares
        try:
            J_damped_pinv = J.T @ np.linalg.inv(J @ J.T + self.lambda_damping * np.eye(6))
            base_joint_velocities = J_damped_pinv @ desired_cartesian_velocity
        except np.linalg.LinAlgError:
            base_joint_velocities = np.zeros(7)
        
        # Compute moment-based corrections
        joint_torques = self.compute_joint_torques()
        cost, moment_error, cost_data = self.compute_base_moment_cost()
        moment_correction = self.compute_null_space_torque_correction(joint_torques)
        
        # Combine base control with moment corrections
        base_joint_velocities_clipped = np.clip(base_joint_velocities, -self.max_joint_velocity, self.max_joint_velocity)
        correction_scaling = 0.01
        final_joint_velocities = base_joint_velocities_clipped + correction_scaling * moment_correction[:7]
        final_joint_velocities = np.clip(final_joint_velocities, -self.max_joint_velocity, self.max_joint_velocity)
        
        # Zero out joint 7 (reserved for manual control)
        #final_joint_velocities[6] = 0.0
        
        # Apply control (uses shared method from BaseController)
        self.apply_control_vector(final_joint_velocities)
        
        # Store for visualization
        self.current_target_orientation = target_orient_mat
        self.previous_position_error = position_error.copy()
        
        return {
            'current_pos': current_pos,
            'current_orient': current_orient_mat,
            'target_orient': target_orient_mat,
            'position_error': position_error,
            'orientation_error': orientation_error,
            'current_joint_vel': current_joint_vel,
            'current_angular_vel': current_angular_vel,
            'moment_cost': cost,
            'moment_error': moment_error,
            'tau_6': cost_data['tau_6'],
            'tau_7': cost_data['tau_7'],
            'projected_moment_to_base': cost_data['projected_moment'],
            'joint_torques': joint_torques
        }


class OldTorqueBalancingController(BaseController):
    """
    Original torque-balancing controller (before refactoring).
    
    This is the original implementation with hardcoded position offsets and
    higher gains, preserved for comparison purposes.
    
    Features:
    - Original hardcoded position offset [-0.15, 0.15, -0.25]
    - Original higher gains (K_pos * 1.0, K_orient * 1.5)
    - tau_7 read from ctrl[6] instead of ctrl[8]
    """
    
    def __init__(self, model, data, **kwargs):
        """Initialize original torque-balancing controller"""
        super().__init__(model, data)
        self.kp_position = kwargs.get('kp_position', 100.0)
        self.kd_position = kwargs.get('kd_position', 20.0)
        self.max_joint_velocity = kwargs.get('max_joint_velocity', 2.0)
        self.target_position = None
        self.target_orientation = None
        self.current_target_orientation = None
        self.torque_balance_gain = kwargs.get('torque_balance_gain', 0.5)
        self.shear_force_threshold = kwargs.get('shear_force_threshold', 0.1)
        
    def compute_joint_torques(self):
        """Compute torques acting on each joint using inverse dynamics."""
        torques = np.zeros(7)
        J = self.compute_jacobian()
        for i in range(7):
            if i < len(self.data.ctrl) and i < len(self.model.actuator_gear):
                gear_ratio = self.model.actuator_gear[i, 0] if len(self.model.actuator_gear[i]) > 0 else 1.0
                torques[i] = self.data.ctrl[i] * gear_ratio
        return torques
    
    def calculate_wrench_at_endeffector(self):
        """Calculate the wrench (force and torque) at the end-effector."""
        wrench = np.zeros(6)
        J = self.compute_jacobian()
        joint_torques = self.compute_joint_torques()
        try:
            J_inv_T = np.linalg.pinv(J.T)
            wrench = J_inv_T @ joint_torques
        except np.linalg.LinAlgError:
            pass
        return wrench
    
    def compute_projected_moment_to_base(self):
        """Compute the moment from the 6th actuator projected to the base."""
        J = self.compute_jacobian()
        J_rot = J[3:6, :7]
        tau_6 = self.data.ctrl[5] if len(self.data.ctrl) > 5 else 0.0
        j6_rot = J_rot[:, 5]
        projected_moment = np.linalg.norm(j6_rot) * np.abs(tau_6)
        return projected_moment, j6_rot, tau_6
    
    def compute_base_moment_cost(self):
        """Compute potential field cost for base moment minimization."""
        # Original: tau_7 from ctrl[6]
        tau_7 = self.data.ctrl[6] if len(self.data.ctrl) > 6 else 0.0
        projected_moment, j6_rot, tau_6 = self.compute_projected_moment_to_base()
        
        moment_error = tau_7 - projected_moment
        cost = 0.5 * moment_error**2
        cost_grad_tau7 = moment_error
        cost_grad_tau6 = -moment_error * np.linalg.norm(j6_rot)
        
        return cost, moment_error, {
            'tau_7': tau_7,
            'tau_6': tau_6,
            'projected_moment': projected_moment,
            'moment_error': moment_error,
            'cost': cost,
            'cost_grad_tau7': cost_grad_tau7,
            'cost_grad_tau6': cost_grad_tau6,
            'j6_rot': j6_rot
        }
    
    def compute_null_space_torque_correction(self, joint_torques):
        """Compute null space torque corrections using gradient of moment cost function."""
        J = self.compute_jacobian()
        try:
            J_pinv = np.linalg.pinv(J)
            N = np.eye(7) - J_pinv @ J
        except np.linalg.LinAlgError:
            N = np.eye(7)
        
        cost, moment_error, cost_data = self.compute_base_moment_cost()
        j6_rot = cost_data['j6_rot']
        
        correction_vector = np.zeros(7)
        if np.abs(moment_error) > 1e-6:
            correction_gain = self.torque_balance_gain
            correction_vector[5] = -moment_error * correction_gain * np.linalg.norm(j6_rot)
            correction_vector[6] = -moment_error * correction_gain
        
        null_space_correction = N @ correction_vector
        self._last_correction_data = {
            'moment_error': moment_error,
            'cost': cost,
            'tau_7': cost_data['tau_7'],
            'tau_6': cost_data['tau_6'],
            'projected_moment': cost_data['projected_moment']
        }
        return null_space_correction
    
    def compute_control(self):
        """Compute torque-balancing control commands (original implementation)"""
        if not self.enabled:
            return
        
        # Get current state
        current_pos = self.get_end_effector_position()
        current_orient_mat = self.get_end_effector_orientation()
        self.target_position = self.get_target_position()
        self.target_orientation = self.get_target_orientation()
        current_joint_vel = self.data.qvel[:7]
        
        # Compute Jacobian and angular velocity
        J_full = self.compute_jacobian()
        J_rot = J_full[3:6, :7]
        current_angular_vel = J_rot @ current_joint_vel
        
        # Original: hardcoded position offset
        position_error = self.target_position - current_pos + np.array([-0.15, 0.15, -0.25])
        
        RotationCorrecter = np.array([[0, -1, 1], [0, 0, -1], [-1, 0, 0]])
        target_orient_mat = RotationCorrecter @ self.target_orientation
        
        R_error = current_orient_mat.T @ target_orient_mat
        orientation_error = np.array([
            R_error[2, 1] - R_error[1, 2],
            R_error[0, 2] - R_error[2, 0],
            R_error[1, 0] - R_error[0, 1]
        ]) * 0.5
        
        target_angular_vel = self.data.qvel[3:6] if self.model.nv > 6 else np.zeros(3)
        angular_vel_err = current_angular_vel - target_angular_vel
        
        J = self.compute_jacobian()
        
        # Original: higher gains
        K_pos = np.diag([8.2, 10.2, 7.0]) * 1.0
        K_angular_vel = np.diag([1.0, 1.0, 1.0]) * 0.05
        K_orient_error = np.diag([5.0, 5.0, 0.0]) * 1.5
        
        desired_position_velocity = K_pos @ position_error
        desired_angular_velocity = (K_orient_error @ orientation_error) - (K_angular_vel @ angular_vel_err)
        desired_cartesian_velocity = np.concatenate([desired_position_velocity, desired_angular_velocity])
        
        try:
            lambda_damping = 0.25
            J_damped_pinv = J.T @ np.linalg.inv(J @ J.T + lambda_damping * np.eye(6))
            base_joint_velocities = J_damped_pinv @ desired_cartesian_velocity
        except np.linalg.LinAlgError:
            base_joint_velocities = np.zeros(7)
        
        joint_torques = self.compute_joint_torques()
        cost, moment_error, cost_data = self.compute_base_moment_cost()
        projected_moment = cost_data['projected_moment']
        tau_6 = cost_data['tau_6']
        tau_7 = cost_data['tau_7']
        
        moment_correction = self.compute_null_space_torque_correction(joint_torques)
        
        # Original: max_velocity = 4.0
        max_velocity = 4.0
        base_joint_velocities_clipped = np.clip(base_joint_velocities, -max_velocity, max_velocity)
        
        correction_scaling = 0.01
        final_joint_velocities = base_joint_velocities_clipped + correction_scaling * moment_correction[:7]
        final_joint_velocities = np.clip(final_joint_velocities, -max_velocity, max_velocity)
        
        # Zero out joint 7 (reserved for manual control)
        final_joint_velocities[6] = 0.0
        
        # Apply control
        self.apply_control_vector(final_joint_velocities)
        
        self.current_target_orientation = target_orient_mat
        self.previous_position_error = position_error.copy()
        
        return {
            'current_pos': current_pos,
            'current_orient': current_orient_mat,
            'target_orient': target_orient_mat,
            'position_error': position_error,
            'orientation_error': orientation_error,
            'current_joint_vel': current_joint_vel,
            'current_angular_vel': current_angular_vel,
            'moment_cost': cost,
            'moment_error': moment_error,
            'tau_6': tau_6,
            'tau_7': tau_7,
            'projected_moment_to_base': projected_moment,
            'joint_torques': joint_torques
        }


class NoControlController(BaseController):
    """
    No-control controller that keeps the robot in a static state with zero control inputs.
    
    This controller is useful for:
    - Testing the dynamics of the free-floating base without active control
    - Observing the robot's behavior under gravity or external forces
    - Baseline testing and validation
    """
    
    ROTATION_CORRECTOR = np.array([[0, -1, 1], [0, 0, -1], [-1, 0, 0]])
    
    def __init__(self, model, data, **kwargs):
        """Initialize no-control controller"""
        super().__init__(model, data)
        self.target_position = None
        self.target_orientation = None
        
    def compute_control(self):
        """Zero all control signals - joints completely unactuated"""
        if not self.enabled:
            return
        
        # Zero all control inputs to ensure robot remains static
        #self.data.ctrl[:] = 0.0
        
        # Get state for monitoring only
        current_pos = self.get_end_effector_position()
        current_orient_mat = self.get_end_effector_orientation()
        self.target_position = self.get_target_position()
        self.target_orientation = self.get_target_orientation()
        current_joint_vel = self.data.qvel[:7]
        
        J = self.compute_jacobian()
        current_angular_vel = J[3:6, :7] @ current_joint_vel
        
        # Use shared methods for error calculation
        position_error = self.compute_position_error(current_pos, self.target_position)
        target_orient_mat = self.ROTATION_CORRECTOR @ self.target_orientation
        orientation_error = self.compute_orientation_error(current_orient_mat, target_orient_mat)
        
        return {
            'current_pos': current_pos,
            'current_orient': current_orient_mat,
            'target_orient': target_orient_mat,
            'position_error': position_error,
            'orientation_error': orientation_error,
            'current_joint_vel': current_joint_vel,
            'current_angular_vel': current_angular_vel
        }


class ControllerFactory(ABC):
    """Factory class for creating controller instances"""
    
    AVAILABLE_CONTROLLERS = {
        'position': PositionController,
        'torque_balancing': TorqueBalancingController,
        'old_torque': OldTorqueBalancingController,
        'no_control': NoControlController,
    }
    
    @classmethod
    def create_controller(cls, controller_type, model, data, **kwargs):
        """
        Create a controller instance
        
        Args:
            controller_type: Type of controller ('position' or 'torque_balancing')
            model: MuJoCo model
            data: MuJoCo data
            **kwargs: Additional parameters for the controller
            
        Returns:
            Controller instance
            
        Raises:
            ValueError: If controller_type is not recognized
        """
        if controller_type not in cls.AVAILABLE_CONTROLLERS:
            raise ValueError(
                f"Unknown controller type: {controller_type}. "
                f"Available: {list(cls.AVAILABLE_CONTROLLERS.keys())}"
            )
        
        controller_class = cls.AVAILABLE_CONTROLLERS[controller_type]
        return controller_class(model, data, **kwargs)
    
    @classmethod
    def get_available_controllers(cls):
        """Get list of available controller types"""
        return list(cls.AVAILABLE_CONTROLLERS.keys())
