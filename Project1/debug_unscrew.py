"""
Debug Unscrewing Simulation
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import os
import sys
import argparse

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wall_crawler_mujoco import WallCrawlerMuJoCoSimulation


class DebugUnscrewSimulation(WallCrawlerMuJoCoSimulation):
    def __init__(self, model_path="dual_arm_robot.xml"):
        super().__init__(model_path)
        self.screw_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "screw")
        self.screw_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "screw_site")
        
        self.logs = {
            "time": [], "screw_pos_x": [], "screw_pos_y": [], "screw_pos_z": [],
            "screw_expected_displacement": [], "screw_actual_displacement": [],
            "ee_pos_x": [], "ee_pos_y": [], "ee_pos_z": [], "ee_to_screw_distance": [],
            "j1": [], "j2": [], "j3": [], "j4": [], "j5": [], "j6": [], "j7": [],
            "j7_velocity": [], "j7_torque": [], "orientation_error": [],
            "compensation_applied": [],  # Track torque compensation
        }
        
        self.screw_wall = None
        self.screw_start_pos = None
        self.screw_axis = None
        self.unscrewing_arm = None
        print("\n🔩 DEBUG UNSCREW SIMULATION INITIALIZED")
    
    def make_screw_visible(self):
        screw_head_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "screw_head")
        if screw_head_id != -1:
            self.model.geom_rgba[screw_head_id] = [1.0, 0.0, 0.0, 1.0]
            self.model.geom_size[screw_head_id] = [0.04, 0.015, 0]
            print("  ✓ Screw head: RED, 4cm radius")
        
        screw_shaft_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "screw_shaft")
        if screw_shaft_id != -1:
            self.model.geom_rgba[screw_shaft_id] = [1.0, 0.5, 0.0, 1.0]
            self.model.geom_size[screw_shaft_id] = [0.015, 0.08, 0]
            self.model.geom_pos[screw_shaft_id] = [0, 0, 0.095]
            print("  ✓ Screw shaft: ORANGE")
        
        screw_thread_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "screw_thread")
        if screw_thread_id != -1:
            self.model.geom_rgba[screw_thread_id] = [1.0, 1.0, 0.0, 0.8]
            self.model.geom_size[screw_thread_id] = [0.018, 0.06, 0]
            self.model.geom_pos[screw_thread_id] = [0, 0, 0.075]
            print("  ✓ Screw thread: YELLOW")
    
    def get_wall_screw_axis(self, wall):
        """Get the axis along which the screw moves when unscrewing (OUT of the wall)."""
        # Unscrewing pulls the screw OUT of the wall (toward the room interior)
        axes = {
            'floor': np.array([0.0, 0.0, 1.0]),      # Floor at Z=0.13, unscrew moves +Z (up)
            'ceiling': np.array([0.0, 0.0, -1.0]),   # Ceiling at Z=2.17, unscrew moves -Z (down)
            'front': np.array([0.0, -1.0, 0.0]),     # Front wall at Y=1.67, unscrew moves -Y (into room)
            'back': np.array([0.0, 1.0, 0.0]),       # Back wall at Y=-0.47, unscrew moves +Y (into room)
            'left_wall': np.array([1.0, 0.0, 0.0]),  # Left wall at X=-2.07, unscrew moves +X (into room)
            'right_wall': np.array([-1.0, 0.0, 0.0]),# Right wall at X=3.87, unscrew moves -X (into room)
        }
        return axes.get(wall, np.array([0.0, 0.0, 1.0]))
    
    def get_screw_orientation_quat(self, wall):
        quats = {
            'floor': [1.0, 0.0, 0.0, 0.0],
            'ceiling': [0.0, 1.0, 0.0, 0.0],
            'back': [0.7071, 0.7071, 0.0, 0.0],
            'front': [0.7071, -0.7071, 0.0, 0.0],
            'left_wall': [0.7071, 0.0, -0.7071, 0.0],
            'right_wall': [0.7071, 0.0, 0.7071, 0.0],
        }
        return np.array(quats.get(wall, [1.0, 0.0, 0.0, 0.0]))
    
    def setup_scenario(self, wall='back', arm='left'):
        print(f"\n🔧 SETTING UP UNSCREW SCENARIO")
        print(f"   Wall: {wall}, Arm: {arm}")
        
        self.screw_wall = wall
        self.unscrewing_arm = arm
        self.screw_axis = self.get_wall_screw_axis(wall)
        
        print("\n📍 Making screw visible...")
        self.make_screw_visible()
        
        # Screw positions on each wall (on the wall surface)
        positions = {
            'back': [0.5, -0.47, 1.2],
            'front': [0.5, 1.67, 1.2],
            'floor': [0.5, 0.5, 0.13],
            'ceiling': [0.5, 0.5, 2.17],
            'left_wall': [-2.07, 0.5, 1.2],
            'right_wall': [3.87, 0.5, 1.2],
        }
        screw_pos = np.array(positions.get(wall, [0.5, -0.47, 1.2]))
        self.screw_start_pos = screw_pos.copy()
        
        self.model.body_pos[self.screw_body_id] = screw_pos
        self.model.body_quat[self.screw_body_id] = self.get_screw_orientation_quat(wall)
        
        print(f"   Screw position: {screw_pos}")
        print(f"   Screw axis: {self.screw_axis}")
        
        # Body position must be where the ARM can reach the screw
        # Left arm points in -X direction (from Y=-0.25 on body)
        # Right arm points in +X direction (from Y=+0.25 on body)
        # Arm reach is ~0.8m, so body should be ~0.5-0.6m away from screw in the arm direction
        
        if wall == 'back':  # Screw at Y=-0.47
            # Body should be in +Y direction from screw, arms point in ±X
            # But arms also bend, so position body so arm can curl to reach -Y
            if arm == 'left':
                # Left arm at Y=-0.25 relative to body, points -X
                # Place body at X offset so arm can reach screw
                body_pos = screw_pos + np.array([0.6, 0.5, 0.0])  # Body in +X,+Y from screw
            else:
                body_pos = screw_pos + np.array([-0.6, 0.5, 0.0])  # Body in -X,+Y from screw
        elif wall == 'front':  # Screw at Y=1.67
            if arm == 'left':
                body_pos = screw_pos + np.array([0.6, -0.5, 0.0])
            else:
                body_pos = screw_pos + np.array([-0.6, -0.5, 0.0])
        elif wall == 'floor':  # Screw at Z=0.13
            body_pos = screw_pos + np.array([0.0, 0.0, 0.6])  # Body above
        elif wall == 'ceiling':  # Screw at Z=2.17
            body_pos = screw_pos + np.array([0.0, 0.0, -0.6])  # Body below
        elif wall == 'left_wall':  # Screw at X=-2.07
            if arm == 'left':
                body_pos = screw_pos + np.array([0.6, 0.0, 0.0])  # Body in +X
            else:
                body_pos = screw_pos + np.array([0.6, 0.0, 0.0])
        elif wall == 'right_wall':  # Screw at X=3.87
            if arm == 'left':
                body_pos = screw_pos + np.array([-0.6, 0.0, 0.0])  # Body in -X
            else:
                body_pos = screw_pos + np.array([-0.6, 0.0, 0.0])
        else:
            body_pos = screw_pos + np.array([0.5, 0.5, 0.0])
        
        self.data.qpos[0:3] = body_pos
        self.data.qpos[3:7] = [1, 0, 0, 0]
        print(f"   Body position: {body_pos}")
        
        mujoco.mj_forward(self.model, self.data)
        
        print("\n⚙️  Solving IK to reach screw...")
        if arm == 'left':
            self.left_target_position = screw_pos.copy()
            # Right arm stays near body
            self.right_target_position = body_pos + np.array([0.3, 0.25, 0.0])
        else:
            self.right_target_position = screw_pos.copy()
            self.left_target_position = body_pos + np.array([0.3, -0.25, 0.0])
        
        # More IK iterations with progress feedback
        best_error = float('inf')
        for i in range(1000):
            self.apply_arm_control_with_orientation('left', wall)
            self.apply_arm_control_with_orientation('right', wall)
            
            # Keep body fixed during IK
            self.data.qpos[0:3] = body_pos
            self.data.qpos[3:7] = [1, 0, 0, 0]
            self.data.qvel[0:6] = 0.0
            
            mujoco.mj_step(self.model, self.data)
            
            current_pos = self.get_left_gripper_pos() if arm == 'left' else self.get_right_gripper_pos()
            error = np.linalg.norm(screw_pos - current_pos)
            
            if error < best_error:
                best_error = error
            
            if error < 0.03:
                print(f"   ✓ IK converged at iter {i}, error: {error*1000:.1f}mm")
                break
                
            if i % 200 == 0 and i > 0:
                print(f"   Iter {i}: error = {error*1000:.1f}mm (best: {best_error*1000:.1f}mm)")
        
        final_error = np.linalg.norm(screw_pos - current_pos)
        if final_error > 0.1:
            print(f"   ⚠️  IK did not fully converge: {final_error*1000:.1f}mm")
        else:
            print(f"   Final error: {final_error*1000:.1f}mm")
        
        print("\n🔒 Locking anchor arm...")
        if arm == 'left':
            self.locked_right_arm_joints = self.data.qpos[self.right_arm_qpos_slice].copy()
            self.right_arm_anchored = True
        else:
            self.locked_left_arm_joints = self.data.qpos[self.left_arm_qpos_slice].copy()
            self.left_arm_anchored = True
        
        self.locked_body_pos = self.data.qpos[0:3].copy()
        self.locked_body_quat = self.data.qpos[3:7].copy()
        
        gripper_idx = self.left_gripper_actuator_idx if arm == 'left' else self.right_gripper_actuator_idx
        self.data.ctrl[gripper_idx] = 255.0
        for _ in range(100):
            mujoco.mj_step(self.model, self.data)
        
        print("\n✅ Setup complete!")
    
    def apply_j7_rotation(self, arm, speed):
        """Apply rotation to J7 using direct position increment - FAST and VISIBLE"""
        if arm == 'left':
            qpos_idx = self.left_arm_qpos_slice.start + 6
            ctrl_idx = self.left_arm_actuator_slice.start + 6
        else:
            qpos_idx = self.right_arm_qpos_slice.start + 6
            ctrl_idx = self.right_arm_actuator_slice.start + 6
        
        # Directly increment the joint position
        current = self.data.qpos[qpos_idx]
        new_val = current + speed * self.model.opt.timestep
        
        # Set both qpos and control target
        self.data.qpos[qpos_idx] = new_val
        self.data.ctrl[ctrl_idx] = new_val
        
        return new_val
    
    def maintain_anchor(self, arm):
        self.data.qpos[0:3] = self.locked_body_pos
        self.data.qpos[3:7] = self.locked_body_quat
        self.data.qvel[0:6] = 0.0
        
        if arm == 'left' and hasattr(self, 'locked_left_arm_joints'):
            self.data.qpos[self.left_arm_qpos_slice] = self.locked_left_arm_joints
            self.data.qvel[self.left_arm_qvel_slice] = 0.0
        elif arm == 'right' and hasattr(self, 'locked_right_arm_joints'):
            self.data.qpos[self.right_arm_qpos_slice] = self.locked_right_arm_joints
            self.data.qvel[self.right_arm_qvel_slice] = 0.0
    
    def apply_torque_compensation(self, arm):
        """Apply gravity/dynamics compensation torques to the arm.
        
        In zero gravity this is minimal, but we add Coriolis/centrifugal compensation
        and damping for smoother motion.
        """
        if arm == 'left':
            qvel_slice = self.left_arm_qvel_slice
            actuator_slice = self.left_arm_actuator_slice
            qpos_slice = self.left_arm_qpos_slice
        else:
            qvel_slice = self.right_arm_qvel_slice
            actuator_slice = self.right_arm_actuator_slice
            qpos_slice = self.right_arm_qpos_slice
        
        # Get current joint velocities
        qvel = self.data.qvel[qvel_slice]
        
        # Apply velocity damping for smooth motion (acts like friction compensation)
        damping_gains = np.array([5.0, 5.0, 5.0, 3.0, 2.0, 1.0, 0.5])  # Nm/(rad/s)
        damping_torques = -damping_gains * qvel
        
        # Get the passive forces (gravity + Coriolis in qfrc_bias)
        # In zero gravity this will mostly be Coriolis/centrifugal forces
        bias_forces = self.data.qfrc_bias[qvel_slice]
        
        # The compensation is to add these bias forces to counteract them
        # (only for joints 1-6, not J7 which we want to rotate freely)
        for i in range(6):
            # Add compensation: current control + bias + damping
            current_ctrl = self.data.ctrl[actuator_slice.start + i]
            # Note: For position-controlled joints, we adjust the target slightly
            # to account for the bias forces
            compensation = bias_forces[i] * 0.001  # Small correction factor
            self.data.ctrl[actuator_slice.start + i] = current_ctrl + compensation
        
        return bias_forces, damping_torques
    
    def maintain_ee_position_and_orientation(self, arm, target_pos):
        """Maintain EE position tracking screw AND strict orientation aligned with screw axis.
        
        The gripper Z-axis should align with the screw axis (pointing into the wall).
        """
        if arm == 'left':
            current_pos = self.get_left_gripper_pos()
            current_orient = self.get_left_ee_orientation()
            site_id = self.left_gripper_site_id
            qpos_slice = self.left_arm_qpos_slice
            actuator_slice = self.left_arm_actuator_slice
        else:
            current_pos = self.get_right_gripper_pos()
            current_orient = self.get_right_ee_orientation()
            site_id = self.right_gripper_site_id
            qpos_slice = self.right_arm_qpos_slice
            actuator_slice = self.right_arm_actuator_slice
        
        # Position error
        pos_error = target_pos - current_pos
        kp_pos = 800.0  # HIGH gain for strict position tracking
        
        # Orientation error - EE Z-axis should align with NEGATIVE screw axis
        # (gripper points INTO the wall, opposite to unscrew direction)
        desired_z = -self.screw_axis  # Gripper Z points into wall
        current_z = current_orient[:, 2]  # Current EE Z-axis
        
        # Compute rotation error using cross product
        orient_error = np.cross(current_z, desired_z)
        kp_orient = 200.0  # Strong orientation control
        
        # Get Jacobians
        jacp, jacr = self.compute_jacobian_at_site(site_id)
        if arm == 'left':
            J_pos = jacp[:, 6:13]
            J_rot = jacr[:, 6:13]
        else:
            J_pos = jacp[:, 13:20]
            J_rot = jacr[:, 13:20]
        
        # Stack position and orientation tasks - POSITION WEIGHTED MORE
        J_full = np.vstack([J_pos * 2.0, J_rot])  # Double weight on position
        task_vel = np.concatenate([kp_pos * pos_error, kp_orient * orient_error])
        
        # Damped least squares with lower regularization for tighter tracking
        lambda_dls = 0.01
        JJT = J_full @ J_full.T
        J_pinv = J_full.T @ np.linalg.inv(JJT + lambda_dls**2 * np.eye(6))
        joint_vel = J_pinv @ task_vel
        
        # Apply to joints 1-6 only (not J7)
        current_q = self.data.qpos[qpos_slice].copy()
        dt = self.model.opt.timestep
        for i in range(6):
            current_q[i] += joint_vel[i] * dt
            self.data.ctrl[actuator_slice.start + i] = current_q[i]
        
        return np.linalg.norm(pos_error), np.linalg.norm(orient_error)
    
    def run_unscrew_task(self, duration=10.0):
        print(f"\n🔩 STARTING UNSCREW")
        print(f"   Duration: {duration}s, Arm: {self.unscrewing_arm}, Wall: {self.screw_wall}")
        
        # MUCH FASTER parameters for visible motion
        unscrew_speed = 6.0  # rad/s - about 1 revolution per second (was 2.0)
        thread_pitch = 0.015  # 15mm per revolution (was 2mm) - MUCH more visible
        max_displacement = 0.15  # 150mm max (was 50mm)
        
        steps = int(duration / self.model.opt.timestep)
        
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data, show_left_ui=True, show_right_ui=True)
        self.viewer.cam.lookat[:] = self.screw_start_pos
        self.viewer.cam.distance = 1.5
        self.viewer.cam.azimuth = 180
        self.viewer.cam.elevation = -20
        
        total_rotation = 0.0
        sim_start = self.data.time
        
        # Get initial J7 position BEFORE any rotation
        if self.unscrewing_arm == 'left':
            j7_start = self.data.qpos[self.left_arm_qpos_slice.start + 6]
        else:
            j7_start = self.data.qpos[self.right_arm_qpos_slice.start + 6]
        
        print("\n   Running...")
        print(f"   Speed: {unscrew_speed:.1f} rad/s ({np.degrees(unscrew_speed):.0f}°/s)")
        print(f"   Thread pitch: {thread_pitch*1000:.1f} mm/rev")
        print(f"   Max displacement: {max_displacement*1000:.0f} mm")
        print(f"   Initial J7: {np.degrees(j7_start):.1f}°")
        
        for step in range(steps):
            if not self.viewer.is_running():
                print("   Viewer closed")
                break
            
            t = self.data.time - sim_start
            
            # Calculate expected displacement based on total rotation
            rotation_inc = unscrew_speed * self.model.opt.timestep
            total_rotation += rotation_inc
            expected_disp = min((total_rotation / (2*np.pi)) * thread_pitch, max_displacement)
            
            # Move screw along its axis
            screw_new_pos = self.screw_start_pos + self.screw_axis * expected_disp
            self.model.body_pos[self.screw_body_id] = screw_new_pos
            
            # Apply J7 rotation (fast and direct)
            j7_val = self.apply_j7_rotation(self.unscrewing_arm, unscrew_speed)
            
            # Maintain EE position AND orientation tracking the screw
            ee_error, orient_error = self.maintain_ee_position_and_orientation(self.unscrewing_arm, screw_new_pos)
            
            # Apply torque compensation for smooth motion
            bias_forces, damping = self.apply_torque_compensation(self.unscrewing_arm)
            
            # Keep anchor arm fixed
            anchor_arm = 'right' if self.unscrewing_arm == 'left' else 'left'
            self.maintain_anchor(anchor_arm)
            
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            
            # Logging
            self.logs["time"].append(t)
            actual_pos = self.data.xpos[self.screw_body_id]
            self.logs["screw_pos_x"].append(actual_pos[0])
            self.logs["screw_pos_y"].append(actual_pos[1])
            self.logs["screw_pos_z"].append(actual_pos[2])
            self.logs["compensation_applied"].append(np.linalg.norm(bias_forces))
            
            actual_disp = np.dot(actual_pos - self.screw_start_pos, self.screw_axis)
            self.logs["screw_expected_displacement"].append(expected_disp)
            self.logs["screw_actual_displacement"].append(actual_disp)
            
            ee_pos = self.get_left_gripper_pos() if self.unscrewing_arm == 'left' else self.get_right_gripper_pos()
            self.logs["ee_pos_x"].append(ee_pos[0])
            self.logs["ee_pos_y"].append(ee_pos[1])
            self.logs["ee_pos_z"].append(ee_pos[2])
            self.logs["ee_to_screw_distance"].append(np.linalg.norm(ee_pos - actual_pos))
            
            qpos_slice = self.left_arm_qpos_slice if self.unscrewing_arm == 'left' else self.right_arm_qpos_slice
            joints = self.data.qpos[qpos_slice]
            for i, jn in enumerate(['j1','j2','j3','j4','j5','j6','j7']):
                self.logs[jn].append(joints[i])
            
            qvel_slice = self.left_arm_qvel_slice if self.unscrewing_arm == 'left' else self.right_arm_qvel_slice
            self.logs["j7_velocity"].append(self.data.qvel[qvel_slice.start + 6])
            self.logs["j7_torque"].append(self.data.qfrc_actuator[qvel_slice.start + 6])
            
            ee_orient = self.get_left_ee_orientation() if self.unscrewing_arm == 'left' else self.get_right_ee_orientation()
            ee_z = ee_orient[:, 2]
            dot = np.abs(np.dot(ee_z, self.screw_axis))
            self.logs["orientation_error"].append(np.degrees(np.arccos(np.clip(dot, -1, 1))))
            
            if step % int(1.0/self.model.opt.timestep) == 0:
                j7_deg = np.degrees(j7_val - j7_start) if j7_start is not None else 0
                print(f"   t={t:.1f}s | J7: {j7_deg:.0f}° | Disp: {expected_disp*1000:.1f}mm | EE err: {ee_error*1000:.1f}mm")
        
        self.viewer.close()
        j7_final = np.degrees(total_rotation)
        print(f"\n✅ UNSCREW COMPLETE")
        print(f"   Total J7 rotation: {j7_final:.0f}° ({j7_final/360:.1f} revolutions)")
        print(f"   Screw displacement: {expected_disp*1000:.1f}mm")
        print(f"   Torque compensation: ACTIVE")
    
    def plot_results(self):
        print("\n📊 Plotting...")
        t = np.array(self.logs["time"])
        if len(t) == 0:
            print("   No data!")
            return
        
        fig, axes = plt.subplots(4, 2, figsize=(14, 16))
        fig.suptitle(f'Unscrew Analysis - Wall: {self.screw_wall}, Arm: {self.unscrewing_arm}', fontsize=14, fontweight='bold')
        
        # Row 0: Screw displacement and EE-to-screw distance
        axes[0,0].plot(t, np.array(self.logs["screw_expected_displacement"])*1000, 'b-', label='Expected', lw=2)
        axes[0,0].plot(t, np.array(self.logs["screw_actual_displacement"])*1000, 'r--', label='Actual', lw=2)
        axes[0,0].set_xlabel('Time (s)'); axes[0,0].set_ylabel('Displacement (mm)')
        axes[0,0].set_title('Screw Displacement Along Axis'); axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)
        
        axes[0,1].plot(t, np.array(self.logs["ee_to_screw_distance"])*1000, 'g-', lw=2)
        axes[0,1].axhline(y=50, color='r', linestyle='--', alpha=0.5, label='50mm threshold')
        axes[0,1].set_xlabel('Time (s)'); axes[0,1].set_ylabel('Distance (mm)')
        axes[0,1].set_title('End-Effector to Screw Distance'); axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)
        
        # Row 1: Joint angles J1-J6 and J7 rotation
        for i, jn in enumerate(['j1','j2','j3','j4','j5','j6']):
            axes[1,0].plot(t, np.degrees(self.logs[jn]), label=f'J{i+1}', lw=1.5)
        axes[1,0].set_xlabel('Time (s)'); axes[1,0].set_ylabel('Angle (deg)')
        axes[1,0].set_title('Joints J1-J6 (should be stable)'); axes[1,0].legend(ncol=3); axes[1,0].grid(True, alpha=0.3)
        
        j7_deg = np.degrees(np.array(self.logs["j7"]))
        j7_rotation = j7_deg - j7_deg[0]  # Relative rotation from start
        axes[1,1].plot(t, j7_rotation, 'purple', lw=2)
        axes[1,1].set_xlabel('Time (s)'); axes[1,1].set_ylabel('Rotation (deg)')
        axes[1,1].set_title(f'J7 Rotation (Total: {j7_rotation[-1]:.0f}°)'); axes[1,1].grid(True, alpha=0.3)
        
        # Row 2: End-effector position XYZ
        axes[2,0].plot(t, np.array(self.logs["ee_pos_x"])*1000, 'r-', label='X', lw=1.5)
        axes[2,0].plot(t, np.array(self.logs["ee_pos_y"])*1000, 'g-', label='Y', lw=1.5)
        axes[2,0].plot(t, np.array(self.logs["ee_pos_z"])*1000, 'b-', label='Z', lw=1.5)
        axes[2,0].set_xlabel('Time (s)'); axes[2,0].set_ylabel('Position (mm)')
        axes[2,0].set_title('End-Effector Position (should follow screw)'); axes[2,0].legend(); axes[2,0].grid(True, alpha=0.3)
        
        # Orientation error
        axes[2,1].plot(t, self.logs["orientation_error"], 'orange', lw=2)
        axes[2,1].axhline(y=10, color='r', linestyle='--', alpha=0.5, label='10° threshold')
        axes[2,1].set_xlabel('Time (s)'); axes[2,1].set_ylabel('Error (deg)')
        axes[2,1].set_title('EE Orientation Error (axis alignment)'); axes[2,1].legend(); axes[2,1].grid(True, alpha=0.3)
        
        # Row 3: J7 velocity and Torque compensation
        axes[3,0].plot(t, np.degrees(self.logs["j7_velocity"]), 'blue', lw=2)
        axes[3,0].set_xlabel('Time (s)'); axes[3,0].set_ylabel('Velocity (deg/s)')
        axes[3,0].set_title('J7 Angular Velocity'); axes[3,0].grid(True, alpha=0.3)
        
        axes[3,1].plot(t, self.logs["compensation_applied"], 'red', lw=2)
        axes[3,1].set_xlabel('Time (s)'); axes[3,1].set_ylabel('Compensation (Nm)')
        axes[3,1].set_title('Torque Compensation Applied'); axes[3,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('unscrew_analysis.png', dpi=150, bbox_inches='tight')
        print("   ✓ Saved: unscrew_analysis.png")
        plt.show()
        
        # Print statistics
        print(f"\n📈 STATISTICS:")
        print(f"   Duration: {t[-1]:.2f}s")
        print(f"   J7 total rotation: {j7_rotation[-1]:.0f}° ({j7_rotation[-1]/360:.1f} revolutions)")
        print(f"   Screw displacement: {self.logs['screw_actual_displacement'][-1]*1000:.1f}mm")
        print(f"   Mean EE-to-screw distance: {np.mean(self.logs['ee_to_screw_distance'])*1000:.1f}mm")
        print(f"   Mean orientation error: {np.mean(self.logs['orientation_error']):.1f}°")
        print(f"   Mean torque compensation: {np.mean(self.logs['compensation_applied']):.3f} Nm")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wall', default='back', choices=['floor','ceiling','front','back','left_wall','right_wall'])
    parser.add_argument('--arm', default='left', choices=['left','right'])
    parser.add_argument('--duration', type=float, default=10.0)
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("       DEBUG UNSCREWING SIMULATION")
    print("="*60)
    
    sim = DebugUnscrewSimulation()
    sim.setup_scenario(wall=args.wall, arm=args.arm)
    sim.run_unscrew_task(duration=args.duration)
    sim.plot_results()
    
    print("\n" + "="*60 + "\n       COMPLETE\n" + "="*60)


if __name__ == "__main__":
    main()
