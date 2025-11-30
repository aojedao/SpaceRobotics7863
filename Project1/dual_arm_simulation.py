#!/usr/bin/env python3
"""
Dual-Arm Zero-Gravity Simulation: Two KUKA iiwa14 arms connected through a central body

This simulation uses two KUKA iiwa14 arms attached to a central box body,
designed for coordinated dual-arm manipulation in zero gravity environments.

Mouse controls are available in the viewer for camera movement.
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import argparse
import matplotlib.pyplot as plt
import os
from scipy.spatial.transform import Rotation
from collections import deque


class DualArmZeroGravitySimulation:
    def __init__(self, 
                 model_path="dual_arm_robot.xml",
                 controller_type='position'):
        """
        Initialize the dual-arm simulation
        
        Args:
            model_path: Path to dual-arm robot model
            controller_type: Type of controller to use ('position' or 'torque_balancing')
        """
        
        # Check if model file exists
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Dual-arm robot model not found: {model_path}")

        # Load the dual-arm model
        print(f"Loading dual-arm model: {model_path}")
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        print("✓ Dual-arm model loaded successfully!")
        
        # Initialize control parameters
        self.setup_control_parameters()
        
        # Store controller type
        self.controller_type = controller_type
        
        # Control state
        self.controller_enabled = True
        self.active_arm = 'left'  # Which arm is currently being controlled
        
        # Target positions for each arm (will be set during initialization)
        self.left_target_position = np.array([0.0, 0.4, 0.5])
        self.right_target_position = np.array([0.0, -0.4, 0.5])
        
        # Target orientations (identity for now)
        self.left_target_orientation = np.eye(3)
        self.right_target_orientation = np.eye(3)
        
        # Controller gains
        self.kp_position = 100.0
        self.kd_position = 20.0
        self.kp_orientation = 50.0
        self.kd_orientation = 10.0
        
        # Movement step sizes
        self.position_step = 0.02  # meters
        self.position_step_fast = 0.05  # faster movement
        
        # Data collection for plotting
        self.max_data_points = 100000
        self.time_data = deque(maxlen=self.max_data_points)
        self.left_position_data = deque(maxlen=self.max_data_points)
        self.right_position_data = deque(maxlen=self.max_data_points)
        self.left_target_data = deque(maxlen=self.max_data_points)
        self.right_target_data = deque(maxlen=self.max_data_points)
        self.left_error_data = deque(maxlen=self.max_data_points)
        self.right_error_data = deque(maxlen=self.max_data_points)
        self.left_joint_data = deque(maxlen=self.max_data_points)
        self.right_joint_data = deque(maxlen=self.max_data_points)
        self.central_body_pos_data = deque(maxlen=self.max_data_points)
        self.central_body_vel_data = deque(maxlen=self.max_data_points)
        self.simulation_start_time = None
        
        print("\n" + "="*70)
        print("🚀 DUAL-ARM ZERO-GRAVITY SIMULATION")
        print("🤖🤖 Two KUKA iiwa14 Arms + Central Body")
        print("="*70)
        print("ROBOT CONFIGURATION:")
        print(f"  • Left Arm: 7 DOF KUKA iiwa14")
        print(f"  • Right Arm: 7 DOF KUKA iiwa14")
        print(f"  • Central Body: Free-floating base (6 DOF)")
        print(f"  • Total DOF: 20 (6 base + 7 left + 7 right)")
        print("CONTROLLER:")
        print(f"  • Active Controller: {controller_type.upper()}")
        print(f"  • Active Arm: {self.active_arm.upper()}")
        print("CONTROLS:")
        print("  • Arrow Keys: Move target position (X/Y)")
        print("  • Page Up/Down: Move target position (Z)")
        print("  • Tab: Switch active arm (Left/Right)")
        print("  • Space: Toggle controller on/off")
        print("  • R: Reset to home position")
        print("  • ESC: Exit simulation")
        print("="*70)

    def setup_control_parameters(self):
        """Setup joint names and control parameters for dual-arm robot"""
        
        # Left arm joints
        self.left_joint_names = [
            "left_joint1", "left_joint2", "left_joint3", "left_joint4",
            "left_joint5", "left_joint6", "left_joint7"
        ]
        
        # Right arm joints
        self.right_joint_names = [
            "right_joint1", "right_joint2", "right_joint3", "right_joint4",
            "right_joint5", "right_joint6", "right_joint7"
        ]
        
        # Get joint IDs
        self.left_joint_ids = []
        for name in self.left_joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.left_joint_ids.append(joint_id)
        
        self.right_joint_ids = []
        for name in self.right_joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.right_joint_ids.append(joint_id)
        
        # Get actuator IDs
        self.left_actuator_ids = []
        for i in range(1, 8):
            act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"left_actuator{i}")
            self.left_actuator_ids.append(act_id)
        
        self.right_actuator_ids = []
        for i in range(1, 8):
            act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"right_actuator{i}")
            self.right_actuator_ids.append(act_id)
        
        # Get body IDs for end-effectors (use gripper base for Jacobian)
        self.left_ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "left_gripper_base_mount")
        self.right_ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "right_gripper_base_mount")
        self.central_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "central_body")
        
        # Get site IDs for gripper centers (for position tracking)
        try:
            self.left_gripper_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_gripper_center")
            self.right_gripper_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_gripper_center")
            print(f"✓ Left gripper site ID: {self.left_gripper_site_id}")
            print(f"✓ Right gripper site ID: {self.right_gripper_site_id}")
        except:
            self.left_gripper_site_id = -1
            self.right_gripper_site_id = -1
        
        # Get site IDs for attachment points
        try:
            self.left_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "left_attachment_site")
            self.right_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "right_attachment_site")
        except:
            self.left_site_id = -1
            self.right_site_id = -1
        
        # Central freejoint
        self.central_freejoint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "central_freejoint")
        
        print(f"✓ Found {len(self.left_joint_ids)} left arm joints")
        print(f"✓ Found {len(self.right_joint_ids)} right arm joints")
        print(f"✓ Found {len(self.left_actuator_ids)} left arm actuators")
        print(f"✓ Found {len(self.right_actuator_ids)} right arm actuators")
        print(f"✓ Left EE body ID: {self.left_ee_body_id}")
        print(f"✓ Right EE body ID: {self.right_ee_body_id}")
        print(f"✓ Central body ID: {self.central_body_id}")

    def get_left_ee_position(self):
        """Get left end-effector position (from gripper center site)"""
        if self.left_gripper_site_id >= 0:
            return self.data.site_xpos[self.left_gripper_site_id].copy()
        return self.data.xpos[self.left_ee_body_id].copy()
    
    def get_right_ee_position(self):
        """Get right end-effector position (from gripper center site)"""
        if self.right_gripper_site_id >= 0:
            return self.data.site_xpos[self.right_gripper_site_id].copy()
        return self.data.xpos[self.right_ee_body_id].copy()
    
    def get_left_ee_orientation(self):
        """Get left end-effector orientation as rotation matrix"""
        if self.left_gripper_site_id >= 0:
            return self.data.site_xmat[self.left_gripper_site_id].reshape(3, 3).copy()
        return self.data.xmat[self.left_ee_body_id].reshape(3, 3).copy()
    
    def get_right_ee_orientation(self):
        """Get right end-effector orientation as rotation matrix"""
        if self.right_gripper_site_id >= 0:
            return self.data.site_xmat[self.right_gripper_site_id].reshape(3, 3).copy()
        return self.data.xmat[self.right_ee_body_id].reshape(3, 3).copy()
    
    def get_central_body_state(self):
        """Get central body position and velocity"""
        pos = self.data.xpos[self.central_body_id].copy()
        # Get velocity from qvel (first 6 elements for freejoint: 3 linear + 3 angular)
        vel = self.data.qvel[:6].copy()
        return pos, vel

    def compute_jacobian(self, body_id):
        """Compute the Jacobian for a given body"""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, body_id)
        return jacp, jacr

    def get_arm_qpos_indices(self, arm='left'):
        """Get qpos indices for arm joints (freejoint uses 7 qpos: 3 pos + 4 quat)"""
        if arm == 'left':
            # After freejoint (7 qpos), left arm starts at qpos[7:14]
            return slice(7, 14)
        else:
            # Right arm after left arm + left gripper: qpos[22:29]
            return slice(22, 29)
    
    def get_arm_qvel_indices(self, arm='left'):
        """Get qvel indices for arm joints (freejoint uses 6 qvel: 3 lin + 3 ang)"""
        if arm == 'left':
            # After freejoint (6 qvel), left arm starts at qvel[6:13]
            return slice(6, 13)
        else:
            # Right arm after left arm + left gripper: qvel[21:28]
            return slice(21, 28)
    
    def get_arm_actuator_indices(self, arm='left'):
        """Get control indices for arm actuators"""
        if arm == 'left':
            return slice(0, 7)
        else:
            return slice(7, 14)

    def compute_arm_control(self, arm='left'):
        """Compute control for specified arm using Cartesian space control"""
        
        if arm == 'left':
            current_pos = self.get_left_ee_position()
            current_orient = self.get_left_ee_orientation()
            target_pos = self.left_target_position
            target_orient = self.left_target_orientation
            body_id = self.left_ee_body_id
            qpos_slice = self.get_arm_qpos_indices('left')
            qvel_slice = self.get_arm_qvel_indices('left')
            actuator_slice = self.get_arm_actuator_indices('left')
        else:
            current_pos = self.get_right_ee_position()
            current_orient = self.get_right_ee_orientation()
            target_pos = self.right_target_position
            target_orient = self.right_target_orientation
            body_id = self.right_ee_body_id
            qpos_slice = self.get_arm_qpos_indices('right')
            qvel_slice = self.get_arm_qvel_indices('right')
            actuator_slice = self.get_arm_actuator_indices('right')
        
        # Position error
        pos_error = target_pos - current_pos
        
        # Orientation error (simplified - using rotation matrix difference)
        orient_error_mat = target_orient @ current_orient.T
        orient_error = Rotation.from_matrix(orient_error_mat).as_rotvec()
        
        # Get Jacobians
        jacp, jacr = self.compute_jacobian(body_id)
        
        # Extract arm-specific Jacobian columns
        # Freejoint: 6 DOF, Left arm: 7 DOF, Left gripper: 8 DOF, Right arm: 7 DOF
        if arm == 'left':
            arm_jacp = jacp[:, 6:13]  # qvel indices 6-12 for left arm
            arm_jacr = jacr[:, 6:13]
        else:
            arm_jacp = jacp[:, 21:28]  # qvel indices 21-27 for right arm
            arm_jacr = jacr[:, 21:28]
        
        # Combine position and orientation Jacobians
        J = np.vstack([arm_jacp, arm_jacr])
        
        # Task space error
        task_error = np.hstack([
            self.kp_position * pos_error,
            self.kp_orientation * orient_error
        ])
        
        # Get current joint velocities
        current_joint_vel = self.data.qvel[qvel_slice]
        
        # Damped least squares inverse
        lambda_dls = 0.1
        JJT = J @ J.T
        J_pinv = J.T @ np.linalg.inv(JJT + lambda_dls**2 * np.eye(6))
        
        # Compute joint velocity command
        joint_vel_cmd = J_pinv @ task_error
        
        # Add joint velocity damping
        joint_vel_cmd -= self.kd_position * current_joint_vel
        
        # Convert to position command (for position-controlled actuators)
        current_joint_pos = self.data.qpos[qpos_slice]
        joint_pos_cmd = current_joint_pos + joint_vel_cmd * self.model.opt.timestep
        
        return joint_pos_cmd, pos_error, orient_error

    def step_controller(self):
        """Step the controller for both arms"""
        
        if not self.controller_enabled:
            return
        
        # Control left arm
        left_cmd, left_pos_error, left_orient_error = self.compute_arm_control('left')
        left_actuator_slice = self.get_arm_actuator_indices('left')
        self.data.ctrl[left_actuator_slice] = left_cmd
        
        # Control right arm
        right_cmd, right_pos_error, right_orient_error = self.compute_arm_control('right')
        right_actuator_slice = self.get_arm_actuator_indices('right')
        self.data.ctrl[right_actuator_slice] = right_cmd
        
        # Collect data
        self.collect_data(left_pos_error, right_pos_error)

    def collect_data(self, left_pos_error, right_pos_error):
        """Collect simulation data for analysis"""
        
        if self.simulation_start_time is None:
            self.simulation_start_time = time.time()
        
        current_time = time.time() - self.simulation_start_time
        
        self.time_data.append(current_time)
        self.left_position_data.append(self.get_left_ee_position())
        self.right_position_data.append(self.get_right_ee_position())
        self.left_target_data.append(self.left_target_position.copy())
        self.right_target_data.append(self.right_target_position.copy())
        self.left_error_data.append(left_pos_error.copy())
        self.right_error_data.append(right_pos_error.copy())
        
        left_slice = self.get_arm_qpos_indices('left')
        right_slice = self.get_arm_qpos_indices('right')
        self.left_joint_data.append(self.data.qpos[left_slice].copy())
        self.right_joint_data.append(self.data.qpos[right_slice].copy())
        
        central_pos, central_vel = self.get_central_body_state()
        self.central_body_pos_data.append(central_pos)
        self.central_body_vel_data.append(central_vel)

    def plot_data(self):
        """Generate comprehensive plots of dual-arm performance"""
        
        if len(self.time_data) < 10:
            print("⚠️ Not enough data collected for plotting")
            return
        
        print("📊 Generating dual-arm analysis plots...")
        
        # Convert to arrays
        time_array = np.array(self.time_data)
        left_pos = np.vstack(self.left_position_data)
        right_pos = np.vstack(self.right_position_data)
        left_target = np.vstack(self.left_target_data)
        right_target = np.vstack(self.right_target_data)
        left_error = np.vstack(self.left_error_data)
        right_error = np.vstack(self.right_error_data)
        left_joints = np.vstack(self.left_joint_data)
        right_joints = np.vstack(self.right_joint_data)
        central_pos = np.vstack(self.central_body_pos_data)
        central_vel = np.vstack(self.central_body_vel_data)
        
        # Create figure with 3x3 subplots
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Dual-Arm KUKA iiwa14 Controller Performance Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Left Arm Position Error Magnitude
        left_error_mag = np.linalg.norm(left_error, axis=1)
        axes[0, 0].plot(time_array, left_error_mag, 'b-', linewidth=2)
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Position Error (m)')
        axes[0, 0].set_title('Left Arm Position Error Magnitude')
        axes[0, 0].grid(True)
        
        # Plot 2: Right Arm Position Error Magnitude
        right_error_mag = np.linalg.norm(right_error, axis=1)
        axes[0, 1].plot(time_array, right_error_mag, 'r-', linewidth=2)
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Position Error (m)')
        axes[0, 1].set_title('Right Arm Position Error Magnitude')
        axes[0, 1].grid(True)
        
        # Plot 3: Both Arms Error Comparison
        axes[0, 2].plot(time_array, left_error_mag, 'b-', label='Left Arm')
        axes[0, 2].plot(time_array, right_error_mag, 'r-', label='Right Arm')
        axes[0, 2].set_xlabel('Time (s)')
        axes[0, 2].set_ylabel('Position Error (m)')
        axes[0, 2].set_title('Both Arms Error Comparison')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # Plot 4: Left Arm Joint Positions
        for i in range(7):
            axes[1, 0].plot(time_array, left_joints[:, i], label=f'Joint {i+1}')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Joint Angle (rad)')
        axes[1, 0].set_title('Left Arm Joint Positions')
        axes[1, 0].legend(loc='upper right', fontsize=8)
        axes[1, 0].grid(True)
        
        # Plot 5: Right Arm Joint Positions
        for i in range(7):
            axes[1, 1].plot(time_array, right_joints[:, i], label=f'Joint {i+1}')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Joint Angle (rad)')
        axes[1, 1].set_title('Right Arm Joint Positions')
        axes[1, 1].legend(loc='upper right', fontsize=8)
        axes[1, 1].grid(True)
        
        # Plot 6: Central Body Position
        axes[1, 2].plot(time_array, central_pos[:, 0], 'r-', label='X')
        axes[1, 2].plot(time_array, central_pos[:, 1], 'g-', label='Y')
        axes[1, 2].plot(time_array, central_pos[:, 2], 'b-', label='Z')
        axes[1, 2].set_xlabel('Time (s)')
        axes[1, 2].set_ylabel('Position (m)')
        axes[1, 2].set_title('Central Body Position')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        
        # Plot 7: Central Body Linear Velocity
        axes[2, 0].plot(time_array, central_vel[:, 0], 'r-', label='Vx')
        axes[2, 0].plot(time_array, central_vel[:, 1], 'g-', label='Vy')
        axes[2, 0].plot(time_array, central_vel[:, 2], 'b-', label='Vz')
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 0].set_ylabel('Velocity (m/s)')
        axes[2, 0].set_title('Central Body Linear Velocity')
        axes[2, 0].legend()
        axes[2, 0].grid(True)
        
        # Plot 8: Central Body Angular Velocity
        axes[2, 1].plot(time_array, central_vel[:, 3], 'r-', label='ωx')
        axes[2, 1].plot(time_array, central_vel[:, 4], 'g-', label='ωy')
        axes[2, 1].plot(time_array, central_vel[:, 5], 'b-', label='ωz')
        axes[2, 1].set_xlabel('Time (s)')
        axes[2, 1].set_ylabel('Angular Velocity (rad/s)')
        axes[2, 1].set_title('Central Body Angular Velocity')
        axes[2, 1].legend()
        axes[2, 1].grid(True)
        
        # Plot 9: 3D Trajectory (Left and Right EE)
        ax3d = fig.add_subplot(3, 3, 9, projection='3d')
        ax3d.plot(left_pos[:, 0], left_pos[:, 1], left_pos[:, 2], 'b-', label='Left EE')
        ax3d.plot(right_pos[:, 0], right_pos[:, 1], right_pos[:, 2], 'r-', label='Right EE')
        ax3d.scatter(left_target[-1, 0], left_target[-1, 1], left_target[-1, 2], 
                     c='b', marker='*', s=100, label='Left Target')
        ax3d.scatter(right_target[-1, 0], right_target[-1, 1], right_target[-1, 2], 
                     c='r', marker='*', s=100, label='Right Target')
        ax3d.set_xlabel('X (m)')
        ax3d.set_ylabel('Y (m)')
        ax3d.set_zlabel('Z (m)')
        ax3d.set_title('End-Effector Trajectories (3D)')
        ax3d.legend()
        
        # Remove the default 2D axes[2, 2] since we replaced with 3D
        axes[2, 2].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('dual_arm_performance_analysis.png', dpi=300, bbox_inches='tight')
        print("📊 Analysis saved to 'dual_arm_performance_analysis.png'!")

    def run_simulation(self, duration=None):
        """Run the dual-arm simulation with viewer"""
        
        if duration is not None:
            print(f"🚀 Starting dual-arm simulation for {duration} seconds...")
        else:
            print("🚀 Starting dual-arm simulation (press ESC to exit)...")
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            start_time = time.time()
            step_count = 0
            
            # Set camera (same as integrated_simulation.py)
            viewer.cam.lookat[:] = [0.0, 0.5, 1.0]
            viewer.cam.distance = 2.249
            viewer.cam.azimuth = 0.75
            viewer.cam.elevation = -20.0
            
            # Reset simulation
            mujoco.mj_resetData(self.model, self.data)
            
            # Set initial position for central body
            self.data.qpos[0] = 0.0   # x
            self.data.qpos[1] = 0.0   # y
            self.data.qpos[2] = 0.5   # z (floating in space)
            self.data.qpos[3] = 1.0   # quaternion w
            self.data.qpos[4] = 0.0   # quaternion x
            self.data.qpos[5] = 0.0   # quaternion y
            self.data.qpos[6] = 0.0   # quaternion z
            
            # Set home position for arms (from keyframe)
            # Left arm (joints qpos[7:14])
            self.data.qpos[7] = 0.0
            self.data.qpos[8] = 0.785398
            self.data.qpos[9] = 0.0
            self.data.qpos[10] = -1.5708
            self.data.qpos[11] = 0.0
            self.data.qpos[12] = 0.0
            self.data.qpos[13] = 0.0
            
            # Left gripper joints (qpos[14:22]) - leave at default
            
            # Right arm (joints qpos[22:29])
            self.data.qpos[22] = 0.0
            self.data.qpos[23] = 0.785398
            self.data.qpos[24] = 0.0
            self.data.qpos[25] = -1.5708
            self.data.qpos[26] = 0.0
            self.data.qpos[27] = 0.0
            self.data.qpos[28] = 0.0
            
            # Right gripper joints (qpos[29:37]) - leave at default
            
            # Zero velocities
            self.data.qvel[:] = 0.0
            self.data.ctrl[:] = 0.0
            
            # Forward kinematics
            mujoco.mj_forward(self.model, self.data)
            
            # Set initial targets to current EE positions
            self.left_target_position = self.get_left_ee_position()
            self.right_target_position = self.get_right_ee_position()
            
            print(f"✓ Dual-arm robot initialized")
            print(f"  Left EE: {self.left_target_position}")
            print(f"  Right EE: {self.right_target_position}")
            
            while viewer.is_running():
                step_start = time.time()
                
                # Run controller
                self.step_controller()
                
                # Step simulation
                mujoco.mj_step(self.model, self.data)
                
                # Update viewer
                viewer.sync()
                
                # Print status every 60 steps
                if step_count % 60 == 0:
                    left_pos = self.get_left_ee_position()
                    right_pos = self.get_right_ee_position()
                    left_error = np.linalg.norm(self.left_target_position - left_pos)
                    right_error = np.linalg.norm(self.right_target_position - right_pos)
                    central_pos, _ = self.get_central_body_state()
                    
                    print(f"Left EE: [{left_pos[0]:.2f}, {left_pos[1]:.2f}, {left_pos[2]:.2f}] | "
                          f"Right EE: [{right_pos[0]:.2f}, {right_pos[1]:.2f}, {right_pos[2]:.2f}] | "
                          f"Errors: L={left_error:.3f}m R={right_error:.3f}m | "
                          f"Base: [{central_pos[0]:.2f}, {central_pos[1]:.2f}, {central_pos[2]:.2f}]")
                
                # Check duration limit
                if duration is not None and (time.time() - start_time) >= duration:
                    break
                
                step_count += 1
                
                # Maintain real-time execution
                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
        
        print("👋 Simulation ended")


def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description='Run the dual-arm zero-gravity simulation')
    parser.add_argument('--duration', '-d', type=float, default=None,
                        help='Duration to run simulation in seconds (default: indefinitely)')
    parser.add_argument('--controller', '-c', type=str, default='position',
                        choices=['position', 'torque_balancing'],
                        help='Controller type (default: position)')
    
    args = parser.parse_args()
    
    simulation = None
    try:
        simulation = DualArmZeroGravitySimulation(controller_type=args.controller)
        simulation.run_simulation(duration=args.duration)
        
        # Generate plots
        print("\n📊 Generating analysis plots...")
        simulation.plot_data()
        
    except KeyboardInterrupt:
        print("\n👋 Simulation interrupted by user")
    except Exception as e:
        print(f"❌ Simulation error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if simulation:
            print("🧹 Cleaning up...")


if __name__ == "__main__":
    main()
