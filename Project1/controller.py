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
    
    @abstractmethod
    def compute_control(self):
        """Compute control commands - must be implemented by subclasses"""
        pass


class PositionController(BaseController):
    """
    Standard position controller using Cartesian space control with quaternion orientation.
    
    Features:
    - 6DOF Cartesian space control
    - Quaternion-based orientation control
    - Redundant manipulator control with null space projection
    - Damped least squares to avoid singularities
    - Angular velocity damping for orientation stability
    """
    
    def __init__(self, model, data, **kwargs):
        """
        Initialize position controller
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            **kwargs: Additional parameters (kp_position, kd_position, max_joint_velocity)
        """
        super().__init__(model, data)
        self.kp_position = kwargs.get('kp_position', 100.0)
        self.kd_position = kwargs.get('kd_position', 20.0)
        self.max_joint_velocity = kwargs.get('max_joint_velocity', 2.0)
        self.target_position = None
        self.target_orientation = None
        self.current_target_orientation = None
    
    def compute_control(self):
        """Compute position control commands"""
        if not self.enabled:
            return
        
        # 1. Get current state
        current_pos = self.get_end_effector_position()
        current_orient_mat = self.get_end_effector_orientation()
        self.target_position = self.get_target_position()
        self.target_orientation = self.get_target_orientation()
        current_joint_vel = self.data.qvel[:7]
        
        # 2. Compute Jacobian and angular velocity
        J_full = self.compute_jacobian()
        J_rot = J_full[3:6, :7]
        current_angular_vel = J_rot @ current_joint_vel
        
        # 3. Calculate position error
        position_error = self.target_position - current_pos + np.array([-0.15, 0.15, -0.25])
        
        # 4. Apply rotation correction
        RotationCorrecter = np.array([[0, -1, 1], [0, 0, -1], [-1, 0, 0]])
        target_orient_mat = RotationCorrecter @ self.target_orientation
        
        # 5. Calculate orientation error
        R_error = current_orient_mat.T @ target_orient_mat
        orientation_error = np.array([
            R_error[2, 1] - R_error[1, 2],
            R_error[0, 2] - R_error[2, 0],
            R_error[1, 0] - R_error[0, 1]
        ]) * 0.5
        
        # 6. Calculate angular velocity error
        target_angular_vel = self.data.qvel[3:6] if self.model.nv > 6 else np.zeros(3)
        angular_vel_err = current_angular_vel - target_angular_vel
        
        # 7. Get Jacobian for control
        J = self.compute_jacobian()
        
        # 8. Controller gains
        K_pos = np.diag([8.2, 10.2, 7.0]) * 1.0
        K_angular_vel = np.diag([1.0, 1.0, 1.0]) * 0.05
        K_orient_error = np.diag([5.0, 5.0, 0.0]) * 1.5
        
        # 9. Compute desired Cartesian velocities
        desired_position_velocity = K_pos @ position_error
        desired_angular_velocity = (K_orient_error @ orientation_error) - (K_angular_vel @ angular_vel_err)
        desired_cartesian_velocity = np.concatenate([desired_position_velocity, desired_angular_velocity])
        
        # 10. Map to joint space with damped least squares
        try:
            lambda_damping = 0.25
            J_damped_pinv = J.T @ np.linalg.inv(J @ J.T + lambda_damping * np.eye(6))
            joint_velocities = J_damped_pinv @ desired_cartesian_velocity
        except np.linalg.LinAlgError:
            joint_velocities = np.zeros(7)
        
        # 11. Apply velocity limits
        max_velocity = 4.0
        joint_velocities_clipped = np.clip(joint_velocities, -max_velocity, max_velocity)
        
        # 12. Zero out joint 7 command (reserved for manual control)
        joint_velocities_clipped[6] = 0.0
        
        # 13. Set control commands (respect manual actuator if present)
        self.apply_control_vector(joint_velocities_clipped)
        
        # Store target orientation for visualization
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

    def apply_control_vector(self, command_vector):
        """Apply a control vector to self.data.ctrl while respecting manual actuator.

        If a manual actuator (named 'manual_joint7') exists its ctrl entry will not be overwritten
        so the viewer slider can be used to inject manual torque into joint7.
        """
        # Ensure data.ctrl has appropriate length
        n_ctrl = len(self.data.ctrl)
        manual_id = getattr(self, 'manual_actuator_id', None)
        #print(f"Manual actuator ID in apply_control_vector: {manual_id}")
        # Apply command vector (skip manual actuator if it exists)
        for i, val in enumerate(command_vector):
            if i < n_ctrl:
                # Skip writing to the manual actuator indexs
                if manual_id is not None and i == manual_id:
                    continue
                self.data.ctrl[i] = float(val)
        
        # Zero remaining ctrl entries (except manual actuator)
        for idx in range(len(command_vector), n_ctrl):
            if manual_id is not None and idx == manual_id:
                # Preserve manual actuator value set by viewer slider
                #self.data.ctrl[8] = 2.0
                continue
            self.data.ctrl[idx] = 0.0


class TorqueBalancingController(BaseController):
    """
    Advanced torque-balancing controller that minimizes base shear forces.
    
    This controller calculates the torque in the sixth axis (wrist rotation) and
    uses the difference between torques in the 6th and 7th axes to minimize
    shear forces transmitted to the base.
    
    Features:
    - Torque calculation and balancing
    - Base shear force minimization
    - Wrench control at end-effector
    - Null space optimization for force distribution
    """
    
    def __init__(self, model, data, **kwargs):
        """
        Initialize torque-balancing controller
        
        Args:
            model: MuJoCo model
            data: MuJoCo data
            **kwargs: Additional parameters
        """
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
        """
        Compute torques acting on each joint using inverse dynamics.
        
        Returns:
            torques: 7-element array of joint torques
        """
        # Compute torques using MuJoCo's inverse dynamics
        torques = np.zeros(7)
        
        # Calculate torques using the relationship tau = J^T * F
        # where J is Jacobian and F is wrench at end effector
        J = self.compute_jacobian()
        
        # Get actuator forces from control and joint properties
        # actuator_gear is a (n_actuators, 6) array, we need the first column (gear ratio)
        for i in range(7):
            if i < len(self.data.ctrl) and i < len(self.model.actuator_gear):
                gear_ratio = self.model.actuator_gear[i, 0] if len(self.model.actuator_gear[i]) > 0 else 1.0
                torques[i] = self.data.ctrl[i] * gear_ratio
        
        return torques
    
    def calculate_wrench_at_endeffector(self):
        """
        Calculate the wrench (force and torque) at the end-effector.
        
        Returns:
            wrench: 6-element array [Fx, Fy, Fz, Mx, My, Mz]
        """
        # In zero-gravity environment, wrench is primarily from control
        # This is a simplified calculation
        wrench = np.zeros(6)
        
        J = self.compute_jacobian()
        joint_torques = self.compute_joint_torques()
        
        # Wrench at end-effector = J^-T * joint_torques
        try:
            J_inv_T = np.linalg.pinv(J.T)
            wrench = J_inv_T @ joint_torques
        except np.linalg.LinAlgError:
            pass
        
        return wrench
    
    def compute_projected_moment_to_base(self):
        """
        Compute the moment from the 6th actuator (wrist rotation) projected to the base.
        
        This represents how the 6th axis torque propagates through the kinematic chain
        and manifests as a moment at the base.
        
        Returns:
            projected_moment: Projected moment magnitude at base from 6th axis torque
        """
        # Get the Jacobian to understand torque transmission
        J = self.compute_jacobian()
        J_rot = J[3:6, :7]  # Rotational part of Jacobian (3x7)
        
        # Get current control/torque in 6th axis
        
        tau_6 = self.data.ctrl[5] if len(self.data.ctrl) > 5 else 0.0
        #print(f"tau_6: {tau_6}")
        
        # The projection of the 6th axis moment to the base is given by
        # how the 6th joint's angular velocity affects the base moment
        # We use the rotational Jacobian row corresponding to axis 6
        
        # Extract the column of J_rot that corresponds to axis 6
        j6_rot = J_rot[:, 5]  # Column 5 (0-indexed) corresponds to joint 6
        #print(f"j6_rot: {j6_rot}")
        
        # The projected moment is the magnitude of this Jacobian column scaled by tau_6
        projected_moment = np.linalg.norm(j6_rot) * np.abs(tau_6)
        
        return projected_moment, j6_rot, tau_6
    
    def compute_base_moment_cost(self):
        """
        Compute potential field cost for base moment minimization.
        
        Cost function: 1/2 * (tau_7 - projected_moment_from_tau_6)^2
        
        This is analogous to obstacle avoidance potential field:
        - tau_7: The 7th axis torque (what we're trying to maintain)
        - projected_moment_from_tau_6: How the 6th axis moment projects to the base
        
        By minimizing their difference, we reduce the net moment at the base.
        
        Returns:
            cost: Potential field cost value
            cost_gradient: Gradient with respect to joint velocities
        """
        # Get 7th axis torque
        tau_7 = self.data.ctrl[8] if len(self.data.ctrl) > 6 else 0.0
        #print(f"tau_7: {tau_7}")
        
        # Get projected moment from 6th axis
        projected_moment, j6_rot, tau_6 = self.compute_projected_moment_to_base()

        
        # Potential field cost: 1/2 * (tau_7 - projected_moment)^2
        moment_error = tau_7 - projected_moment
        cost = 0.5 * moment_error**2
        
        # Gradient of cost with respect to the moment difference
        # For null-space correction, we need to understand how to reduce this cost
        # Cost gradient w.r.t. torques: d(cost)/d(tau_7) and d(cost)/d(tau_6)
        cost_grad_tau7 = moment_error  # Positive if tau_7 > projected_moment
        cost_grad_tau6 = -moment_error * np.linalg.norm(j6_rot)  # How tau_6 affects cost
        
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
        """
        Compute null space torque corrections using gradient of moment cost function.
        
        This method computes corrections in the null space of the Jacobian that
        minimize the potential field cost: 1/2 * (tau_7 - projected_moment_from_tau_6)^2
        
        The correction moves in the direction that reduces the moment error while
        maintaining the primary end-effector task.
        
        Args:
            joint_torques: 7-element array of current joint torques
            
        Returns:
            correction_torques: Corrective torques in null space to minimize base moment
        """
        J = self.compute_jacobian()
        
        # Compute null space projector: N = I - J^+ * J
        try:
            J_pinv = np.linalg.pinv(J)
            N = np.eye(7) - J_pinv @ J
        except np.linalg.LinAlgError:
            N = np.eye(7)
        
        # Get moment cost information
        cost, moment_error, cost_data = self.compute_base_moment_cost()
        
        # Extract gradient information
        cost_grad_tau7 = cost_data['cost_grad_tau7']
        cost_grad_tau6 = cost_data['cost_grad_tau6']
        j6_rot = cost_data['j6_rot']
        tau_6 = cost_data['tau_6']
        projected_moment = cost_data['projected_moment']
        
        # Create a correction vector that reduces the cost
        # We want to adjust joint velocities to minimize:
        # 1/2 * (tau_7 - projected_moment)^2
        
        # Correction strategy:
        # If tau_7 > projected_moment: we want to reduce tau_7 or increase projected_moment
        # If tau_7 < projected_moment: we want to increase tau_7 or decrease projected_moment
        
        correction_vector = np.zeros(7)
        
        # For joint 6 (index 5): affects projected_moment through tau_6
        # Reducing tau_6 reduces projected_moment (if tau_7 > projected_moment)
        if np.abs(moment_error) > 1e-6:  # Only correct if there's significant error
            # Gain for the correction (tuned parameter)
            correction_gain = self.torque_balance_gain
            
            # Joint 6 correction: proportional to moment error
            # If moment_error > 0 (tau_7 > projected_moment), reduce tau_6
            correction_vector[5] = -moment_error * correction_gain * np.linalg.norm(j6_rot)
            
            # Joint 7 correction: proportional to moment error (opposite sign)
            # If moment_error > 0 (tau_7 > projected_moment), reduce tau_7
            correction_vector[6] = -moment_error * correction_gain
        
        # Project correction to null space to not disturb primary task
        null_space_correction = N @ correction_vector
        
        # Store debug info for later analysis
        self._last_correction_data = {
            'moment_error': moment_error,
            'cost': cost,
            'tau_7': cost_data['tau_7'],
            'tau_6': cost_data['tau_6'],
            'projected_moment': projected_moment
        }
        
        return null_space_correction
    
    def compute_control(self):
        """Compute torque-balancing control commands using moment minimization"""
        if not self.enabled:
            return
        
        # 1. Get current state (same as position controller)
        current_pos = self.get_end_effector_position()
        current_orient_mat = self.get_end_effector_orientation()
        self.target_position = self.get_target_position()
        self.target_orientation = self.get_target_orientation()
        current_joint_vel = self.data.qvel[:7]
        
        # 2. Compute Jacobian and angular velocity
        J_full = self.compute_jacobian()
        J_rot = J_full[3:6, :7]
        current_angular_vel = J_rot @ current_joint_vel
        
        # 3. Calculate position and orientation errors
        position_error = self.target_position - current_pos + np.array([-0.15, 0.15, -0.25])
        
        RotationCorrecter = np.array([[0, -1, 1], [0, 0, -1], [-1, 0, 0]])
        target_orient_mat = RotationCorrecter @ self.target_orientation
        
        R_error = current_orient_mat.T @ target_orient_mat
        orientation_error = np.array([
            R_error[2, 1] - R_error[1, 2],
            R_error[0, 2] - R_error[2, 0],
            R_error[1, 0] - R_error[0, 1]
        ]) * 0.5
        
        # 4. Compute base position control (same as position controller)
        target_angular_vel = self.data.qvel[3:6] if self.model.nv > 6 else np.zeros(3)
        angular_vel_err = current_angular_vel - target_angular_vel
        
        J = self.compute_jacobian()
        
        K_pos = np.diag([8.2, 10.2, 7.0]) * 1.0
        K_angular_vel = np.diag([1.0, 1.0, 1.0]) * 0.05
        K_orient_error = np.diag([5.0, 5.0, 0.0]) * 1.5
        
        desired_position_velocity = K_pos @ position_error
        desired_angular_velocity = (K_orient_error @ orientation_error) - (K_angular_vel @ angular_vel_err)
        desired_cartesian_velocity = np.concatenate([desired_position_velocity, desired_angular_velocity])
        
        # 5. Compute base position control velocities
        try:
            lambda_damping = 0.25
            J_damped_pinv = J.T @ np.linalg.inv(J @ J.T + lambda_damping * np.eye(6))
            base_joint_velocities = J_damped_pinv @ desired_cartesian_velocity
        except np.linalg.LinAlgError:
            base_joint_velocities = np.zeros(7)
        
        # 6. Compute joint torques
        joint_torques = self.compute_joint_torques()
        
        # 7. Compute base moment cost using potential field function
        cost, moment_error, cost_data = self.compute_base_moment_cost()
        projected_moment = cost_data['projected_moment']
        tau_6 = cost_data['tau_6']
        tau_7 = cost_data['tau_7']
        
        # 8. Compute null space moment correction using gradient descent
        moment_correction = self.compute_null_space_torque_correction(joint_torques)
        
        # 9. Combine base position control with moment balancing
        # Use velocity commands for primary task, moment corrections in null space
        max_velocity = 4.0
        base_joint_velocities_clipped = np.clip(base_joint_velocities, -max_velocity, max_velocity)
        
        # 10. Apply corrections (small corrections to avoid task disruption)
        correction_scaling = 0.01  # Small scaling factor for moment corrections
        final_joint_velocities = base_joint_velocities_clipped + correction_scaling * moment_correction[:7]
        final_joint_velocities = np.clip(final_joint_velocities, -max_velocity, max_velocity)
        
        # 11. Zero out joint 7 command (reserved for manual control)
        final_joint_velocities[6] = 0.0
        
        # 12. Set control commands (respect manual actuator if present)
        # Use apply_control_vector helper to avoid overwriting manual actuator slider
        if hasattr(self, 'apply_control_vector'):
            self.apply_control_vector(final_joint_velocities)
        else:
            # Fallback: manually apply control while protecting manual actuator
            manual_id = getattr(self, 'manual_actuator_id', None)
            manual_id = 6
            for i, val in enumerate(final_joint_velocities):
                if i < len(self.data.ctrl):
                    if manual_id is not None and i == manual_id:
                        continue  # Skip manual actuator
                    self.data.ctrl[i] = float(val)
            # Zero remaining control entries (except manual actuator)
            for idx in range(len(final_joint_velocities), len(self.data.ctrl)):
                if manual_id is not None and idx == manual_id:
                    continue
                #print(f"Zeroing ctrl at index {idx}")
                self.data.ctrl[idx] = 0.0
        
        # Store target orientation
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


class ControllerFactory:
    """Factory class for creating controller instances"""
    
    AVAILABLE_CONTROLLERS = {
        'position': PositionController,
        'torque_balancing': TorqueBalancingController,
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
