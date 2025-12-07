
import numpy as np
import mujoco
from enums import GripperState
import config

# ============================================================================
# CONTROLLER MIXIN
# ============================================================================

class ControllerMixin:
    """Methods for arm control, kinematics, and gripper management"""

    # ------------------------------------------------------------------------
    # State Accessors
    # ------------------------------------------------------------------------
    
    def get_left_gripper_pos(self) -> np.ndarray:
        """Get left gripper center position"""
        return self.data.site_xpos[self.left_gripper_site_id].copy()
    
    def get_right_gripper_pos(self) -> np.ndarray:
        """Get right gripper center position"""
        return self.data.site_xpos[self.right_gripper_site_id].copy()
    
    def get_central_body_pos(self) -> np.ndarray:
        """Get central body position"""
        return self.data.xpos[self.central_body_id].copy()
    
    def get_left_link4_pos(self) -> np.ndarray:
        """Get left link4 position (elbow - workspace sphere center)"""
        return self.data.xpos[self.left_link4_id].copy()
    
    def get_right_link4_pos(self) -> np.ndarray:
        """Get right link4 position (elbow - workspace sphere center)"""
        return self.data.xpos[self.right_link4_id].copy()
        
    def get_left_ee_orientation(self) -> np.ndarray:
        """Get left end-effector orientation as 3x3 rotation matrix"""
        return self.data.site_xmat[self.left_gripper_site_id].reshape(3, 3).copy()
    
    def get_right_ee_orientation(self) -> np.ndarray:
        """Get right end-effector orientation as 3x3 rotation matrix"""
        return self.data.site_xmat[self.right_gripper_site_id].reshape(3, 3).copy()

    # ------------------------------------------------------------------------
    # Geometric Helpers
    # ------------------------------------------------------------------------

    def get_surface_normal(self, wall: str) -> np.ndarray:
        """Get the normal vector for a wall surface (pointing OUT of surface).
        Note: WallPositionMapper returns INWARD normals. This returns standard normals.
        """
        normals = {
            'floor': np.array([0.0, 0.0, 1.0]),       # Up (from floor)
            'ceiling': np.array([0.0, 0.0, -1.0]),    # Down (from ceiling)
            'front': np.array([0.0, -1.0, 0.0]),      # -Y (from front +Y wall)
            'back': np.array([0.0, 1.0, 0.0]),        # +Y (from back -Y wall)
            'left_wall': np.array([1.0, 0.0, 0.0]),   # +X (from left -X wall)
            'right_wall': np.array([-1.0, 0.0, 0.0]), # -X (from right +X wall)
        }
        return normals.get(wall, np.array([0.0, 0.0, -1.0]))  # Default: down

    def get_target_orientation_matrix(self, wall: str) -> np.ndarray:
        """Get desired end-effector orientation matrix for approaching a wall."""
        # Surface normal is direction gripper should point (its Z-axis)
        z_axis = -self.get_surface_normal(wall)  # Gripper points INTO surface
        
        # Choose an arbitrary up vector (avoid parallel with z_axis)
        if abs(z_axis[2]) < 0.9:
            up = np.array([0.0, 0.0, 1.0])
        else:
            up = np.array([0.0, 1.0, 0.0])
        
        # Compute orthonormal basis
        x_axis = np.cross(up, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        # Build rotation matrix [x, y, z] as columns
        R = np.column_stack([x_axis, y_axis, z_axis])
        return R

    def compute_jacobian(self, body_id: int):
        """Compute the Jacobian for a given body (at body origin)"""
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        mujoco.mj_jacBody(self.model, self.data, jacp, jacr, body_id)
        return jacp, jacr

    def compute_jacobian_at_site(self, site_id: int):
        """Compute the Jacobian at a specific site position (e.g., gripper center).
        
        This is more accurate than compute_jacobian for end-effector control
        because it accounts for the site offset from the body origin.
        """
        jacp = np.zeros((3, self.model.nv))
        jacr = np.zeros((3, self.model.nv))
        # Get the site position in world frame
        site_pos = self.data.site_xpos[site_id]
        # Get the body that the site is attached to
        body_id = self.model.site_bodyid[site_id]
        # Compute Jacobian at the site position (not body origin)
        mujoco.mj_jac(self.model, self.data, jacp, jacr, site_pos, body_id)
        return jacp, jacr

    # ------------------------------------------------------------------------
    # Control & Actuation
    # ------------------------------------------------------------------------

    def set_gripper(self, arm: str, close: bool):
        """Set gripper state (open or closed)"""
        value = self.gripper_closed_value if close else self.gripper_open_value
        
        if arm == 'left':
            self.data.ctrl[self.left_gripper_actuator_idx] = value
            self.left_gripper_state = GripperState.CLOSED if close else GripperState.OPEN
            if not close:
                self.release_anchor('left')
        else:
            self.data.ctrl[self.right_gripper_actuator_idx] = value
            self.right_gripper_state = GripperState.CLOSED if close else GripperState.OPEN
            if not close:
                self.release_anchor('right')
        
        state_str = "CLOSED" if close else "OPEN"
        print(f"  🤏 {arm.upper()} gripper: {state_str}")

    def get_gripper_closure(self, arm: str) -> float:
        """Get how closed the gripper is (0.0 = fully open, 1.0 = fully closed)"""
        if arm == 'left':
            ctrl_value = self.data.ctrl[self.left_gripper_actuator_idx]
        else:
            ctrl_value = self.data.ctrl[self.right_gripper_actuator_idx]
        
        closure = ctrl_value / self.gripper_closed_value
        return np.clip(closure, 0.0, 1.0)
    
    def is_gripper_closed_enough(self, arm: str, threshold: float = 0.8) -> bool:
        """Check if gripper is closed enough to maintain anchor"""
        return self.get_gripper_closure(arm) >= threshold

    def release_anchor(self, arm: str):
        """Release the anchor for specified arm"""
        if arm == 'left':
            if self.left_arm_anchored:
                print(f"  🔓 {arm.upper()} arm RELEASED from anchor")
            self.left_arm_anchored = False
            self.left_anchor_position = None
        else:
            if self.right_arm_anchored:
                print(f"  🔓 {arm.upper()} arm RELEASED from anchor")
            self.right_arm_anchored = False
            self.right_anchor_position = None

    def detect_arm_crossing(self, verbose: bool = False):
        """Detect if the arms are crossing each other."""
        body_pos = self.get_central_body_pos()
        body_quat = self.data.qpos[3:7]
        
        left_grip = self.get_left_gripper_pos()
        right_grip = self.get_right_gripper_pos()
        
        # Get body yaw
        w, x, y, z = body_quat
        body_yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        
        cos_yaw = np.cos(-body_yaw)
        sin_yaw = np.sin(-body_yaw)
        
        left_rel = left_grip - body_pos
        right_rel = right_grip - body_pos
        
        left_local_x = left_rel[0] * cos_yaw - left_rel[1] * sin_yaw
        right_local_x = right_rel[0] * cos_yaw - right_rel[1] * sin_yaw
        
        left_on_wrong_side = left_local_x > 0.1
        right_on_wrong_side = right_local_x < -0.1
        
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
        
        if is_crossing and verbose and config.PRINT_CROSSING_DIAGNOSTICS:
            diag_counter = getattr(self, '_crossing_diag_counter', 0) + 1
            self._crossing_diag_counter = diag_counter
            if diag_counter % 200 == 1:
                print(f"\n  ╔══════════════════════════════════════════════════════════╗")
                print(f"  ║           ⚠️  ARM CROSSING DIAGNOSTICS                    ║")
                print(f"  ╠══════════════════════════════════════════════════════════╣")
                print(f"  ║ Body Position: ({body_pos[0]:.3f}, {body_pos[1]:.3f}, {body_pos[2]:.3f})")
                print(f"  ║ Body Yaw: {np.degrees(body_yaw):.1f}°")
                print(f"  ╟──────────────────────────────────────────────────────────╢")
                print(f"  ║ LEFT Gripper (world):  ({left_grip[0]:.3f}, {left_grip[1]:.3f}, {left_grip[2]:.3f})")
                print(f"  ║ LEFT Gripper (local):  X={left_local_x:+.3f}, Y={left_rel[0] * sin_yaw + left_rel[1] * cos_yaw:+.3f}")
                print(f"  ║   → Should be X < 0.1 (left side) | {'❌ ON WRONG SIDE' if left_on_wrong_side else '✓ OK'}")
                print(f"  ╟──────────────────────────────────────────────────────────╢")
                print(f"  ║ RIGHT Gripper (world): ({right_grip[0]:.3f}, {right_grip[1]:.3f}, {right_grip[2]:.3f})")
                print(f"  ║ RIGHT Gripper (local): X={right_local_x:+.3f}, Y={right_rel[0] * sin_yaw + right_rel[1] * cos_yaw:+.3f}")
                print(f"  ║   → Should be X > -0.1 (right side) | {'❌ ON WRONG SIDE' if right_on_wrong_side else '✓ OK'}")
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

    def compute_arm_control(self, arm: str = 'left', anchored_mode: bool = False, skip_base_joints: int = 0):
        """Compute Cartesian space control for specified arm"""
        if arm == 'left':
            current_pos = self.get_left_gripper_pos()
            target_pos = self.left_target_position
            site_id = self.left_gripper_site_id  # Use site for accurate Jacobian
            qpos_slice = self.left_arm_qpos_slice
            qvel_slice = self.left_arm_qvel_slice
        else:
            current_pos = self.get_right_gripper_pos()
            target_pos = self.right_target_position
            site_id = self.right_gripper_site_id  # Use site for accurate Jacobian
            qpos_slice = self.right_arm_qpos_slice
            qvel_slice = self.right_arm_qvel_slice
        
        if target_pos is None:
            return self.data.qpos[qpos_slice].copy(), np.zeros(3)
        
        pos_error = target_pos - current_pos
        # Use site-based Jacobian for accurate gripper tip control
        jacp, jacr = self.compute_jacobian_at_site(site_id)
        
        if arm == 'left':
            arm_jacp = jacp[:, 6:13]
        else:
            arm_jacp = jacp[:, 21:28]
        
        J = arm_jacp
        task_error = self.kp_position * pos_error
        
        current_joint_vel = self.data.qvel[qvel_slice]
        
        JJT = J @ J.T
        J_pinv = J.T @ np.linalg.inv(JJT + self.lambda_dls**2 * np.eye(3))
        
        joint_vel_cmd = J_pinv @ task_error
        joint_vel_cmd = joint_vel_cmd * self.joint_gain_scale
        joint_vel_cmd -= self.kd_position * current_joint_vel
        
        current_joint_pos = self.data.qpos[qpos_slice]
        joint_pos_cmd = current_joint_pos + joint_vel_cmd * self.model.opt.timestep
        
        # Joint 1 prioritization logic (keep arms on correct sides)
        arm_is_anchored = (arm == 'left' and self.left_arm_anchored) or \
                          (arm == 'right' and self.right_arm_anchored)
        
        is_crossing, crossing_severity = self.detect_arm_crossing(verbose=True)
        
        # DISABLED: Joint 1 crossing correction was causing arms to get stuck
        # The recovery logic in the main simulation handles arm crossing better
        # by using intelligent IK targeting instead of fighting the IK solution
        
        # Apply skip_base_joints logic
        if anchored_mode:
            skip_base_joints = max(skip_base_joints, 4)  # Lock first 5 joints if anchored
            
        if skip_base_joints > 0:
                    joint_pos_cmd[:skip_base_joints] = current_joint_pos[:skip_base_joints]
                    
                    # Also adjust the other (anchored) arm's joints
                    other_arm = 'right' if arm == 'left' else 'left'
                    if other_arm == 'left':
                        other_qpos_slice = self.left_arm_qpos_slice
                        other_actuator_slice = self.left_arm_actuator_slice
                    else:
                        other_qpos_slice = self.right_arm_qpos_slice
                        other_actuator_slice = self.right_arm_actuator_slice
                    
                    other_current_joint_pos = self.data.qpos[other_qpos_slice]
                    
                    self.data.ctrl[other_actuator_slice][:skip_base_joints] = other_current_joint_pos[:skip_base_joints]
            
        return joint_pos_cmd, pos_error

    def apply_arm_control(self, arm: str = 'left', anchored_mode: bool = False, skip_base_joints: int = 0):
        """Apply computed control to arm"""
        joint_pos_cmd, pos_error = self.compute_arm_control(arm, anchored_mode, skip_base_joints)
        
        if arm == 'left':
            self.data.ctrl[self.left_arm_actuator_slice] = joint_pos_cmd
        else:
            self.data.ctrl[self.right_arm_actuator_slice] = joint_pos_cmd
            
        return np.linalg.norm(pos_error)

    def apply_arm_control_with_orientation(self, arm: str, target_wall: str, skip_base_joints: int = 0):
        """Apply arm control with orientation alignment to approach wall perpendicularly.
        
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
            site_id = self.left_gripper_site_id  # Use site for accurate Jacobian
            qpos_slice = self.left_arm_qpos_slice
            qvel_slice = self.left_arm_qvel_slice
            actuator_slice = self.left_arm_actuator_slice
        else:
            current_pos = self.get_right_gripper_pos()
            current_orient = self.get_right_ee_orientation()
            target_pos = self.right_target_position
            site_id = self.right_gripper_site_id  # Use site for accurate Jacobian
            qpos_slice = self.right_arm_qpos_slice
            qvel_slice = self.right_arm_qvel_slice
            actuator_slice = self.right_arm_actuator_slice
        
        if target_pos is None:
            return 0.0
        
        # Position error
        pos_error = target_pos - current_pos
        
        # Orientation error - align gripper Z-axis with surface approach direction
        target_orient = self.get_target_orientation_matrix(target_wall)
        orient_error_matrix = target_orient @ current_orient.T
        
        # Extract axis-angle from rotation matrix for orientation error
        trace = np.trace(orient_error_matrix)
        angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
        
        if angle > 1e-6:
            # Extract rotation axis
            axis = np.array([
                orient_error_matrix[2, 1] - orient_error_matrix[1, 2],
                orient_error_matrix[0, 2] - orient_error_matrix[2, 0],
                orient_error_matrix[1, 0] - orient_error_matrix[0, 1]
            ])
            axis = axis / (2 * np.sin(angle) + 1e-10)
            orient_error = angle * axis
        else:
            orient_error = np.zeros(3)
        
        # Get full Jacobian at gripper site (not body origin)
        jacp, jacr = self.compute_jacobian_at_site(site_id)
        
        # Extract arm-specific Jacobians
        if arm == 'left':
            arm_jacp = jacp[:, 6:13]
            arm_jacr = jacr[:, 6:13]
        else:
            arm_jacp = jacp[:, 21:28]
            arm_jacr = jacr[:, 21:28]
        
        # Stack position and orientation Jacobians (6x7)
        J = np.vstack([arm_jacp, arm_jacr])
        
        # Task error: position + scaled orientation
        task_error = np.concatenate([
            self.kp_position * pos_error,
            self.kp_orientation * orient_error
        ])
        
        # Current joint velocities
        current_joint_vel = self.data.qvel[qvel_slice]
        
        # Damped least squares for 6x7 Jacobian
        JJT = J @ J.T  # 6x6
        J_pinv = J.T @ np.linalg.inv(JJT + self.lambda_dls**2 * np.eye(6))  # 7x6
        
        # Compute joint velocity command
        joint_vel_cmd = J_pinv @ task_error
        
        # Add velocity damping
        joint_vel_cmd -= self.kd_position * current_joint_vel
        
        # Convert to position command
        current_joint_pos = self.data.qpos[qpos_slice]
        joint_pos_cmd = current_joint_pos + joint_vel_cmd * self.model.opt.timestep
        
        # Apply skip_base_joints if needed
        if skip_base_joints > 0:
            if arm == 'left':
                current_base = self.data.qpos[7:7+skip_base_joints]
            else:
                current_base = self.data.qpos[22:22+skip_base_joints]
            joint_pos_cmd[:skip_base_joints] = current_base
        
        # Apply control
        self.data.ctrl[actuator_slice] = joint_pos_cmd
        
        return np.linalg.norm(pos_error)

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

    def maintain_anchor(self, arm: str, control_anchored_arm: bool = False):
        """Maintain the anchor by LOCKING body position"""
        if arm == 'left' and self.left_arm_anchored:
            if not self.is_gripper_closed_enough('left'):
                self.release_anchor('left')
                return 0.0
            
            # Restore body position to locked position
            if hasattr(self, 'locked_body_pos') and self.locked_body_pos is not None:
                self.data.qpos[0:3] = self.locked_body_pos.copy()
                self.data.qpos[3:7] = self.locked_body_quat.copy()
                self.data.qvel[0:6] = 0.0  # Zero body velocity
                
                # Also restore left arm joints to keep gripper at anchor
                if hasattr(self, 'locked_left_arm_joints') and self.locked_left_arm_joints is not None:
                    self.data.qpos[7:14] = self.locked_left_arm_joints.copy()
                    self.data.qvel[6:13] = 0.0  # Zero left arm velocities
                
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
            
            if hasattr(self, 'locked_body_pos') and self.locked_body_pos is not None:
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
        """Lock the current body position specified for anchor constraint."""
        self.locked_body_pos = self.data.qpos[0:3].copy()
        self.locked_body_quat = self.data.qpos[3:7].copy()
        self.locked_left_arm_joints = self.data.qpos[7:14].copy()
        print(f"  🔒 Body locked at: ({self.locked_body_pos[0]:.2f}, {self.locked_body_pos[1]:.2f}, {self.locked_body_pos[2]:.2f})")
        left_grip = self.get_left_gripper_pos()
        print(f"  🔒 Left gripper at: ({left_grip[0]:.2f}, {left_grip[1]:.2f}, {left_grip[2]:.2f})")

    def apply_coordinated_arm_control(self, anchored_arm: str, moving_target: np.ndarray, anchor_pos: np.ndarray):
        """Use first 5 joints of anchored arm to help body reach moving arm's target."""
        # Get current positions
        body_pos = self.get_central_body_pos()
        body_to_target = moving_target - body_pos
        target_distance = np.linalg.norm(body_to_target)
        
        # Don't bother if body is already very close to target
        if target_distance < 0.2:
            return
        
        # Get arm-specific data
        if anchored_arm == 'left':
            site_id = self.left_gripper_site_id
            qpos_slice = self.left_arm_qpos_slice
            qvel_slice = self.left_arm_qvel_slice
            actuator_slice = self.left_arm_actuator_slice
        else:
            site_id = self.right_gripper_site_id
            qpos_slice = self.right_arm_qpos_slice
            qvel_slice = self.right_arm_qvel_slice
            actuator_slice = self.right_arm_actuator_slice
        
        # Desired body motion direction
        body_direction = body_to_target / target_distance
        desired_body_vel = body_direction * min(target_distance, 0.8)
        
        # Get the Jacobian at gripper site (more accurate than body origin)
        jacp_full, _ = self.compute_jacobian_at_site(site_id)
        
        # Extract arm-specific Jacobian for first 5 joints only
        if anchored_arm == 'left':
            J_base = jacp_full[:, 6:11]
        else:
            J_base = jacp_full[:, 21:26]
        
        kp_coord = 2500.0
        kp_repulsive = 0.0
        
        combined_body_vel = (kp_coord * desired_body_vel) + (kp_repulsive * (-body_direction))
        
        # So final task input to pinv
        task_vel = -combined_body_vel
        
        # Damped least squares inverse
        lambda_coord = 0.08
        JJT = J_base @ J_base.T
        J_pinv = J_base.T @ np.linalg.inv(JJT + lambda_coord**2 * np.eye(3))
        
        # Joint velocities
        joint_vel_base = J_pinv @ task_vel
        
        # Apply damping
        kd_coord = 20.0
        current_joint_vel = self.data.qvel[qvel_slice][:5]
        joint_vel_base -= kd_coord * current_joint_vel
        
        # Convert to position command for first 5 joints
        current_q = self.data.qpos[qpos_slice]
        dt = self.model.opt.timestep
        
        for i in range(5):
            new_q = current_q[i] + joint_vel_base[i] * dt
            self.data.ctrl[actuator_slice.start + i] = new_q
