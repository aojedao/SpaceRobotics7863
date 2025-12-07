
import numpy as np
import mujoco
from enums import RobotState

class TaskMixin:
    """Methods related to specific tasks like unscrewing"""

    def _execute_unscrew_control(self, unscrewing_arm: str, anchored_arm: str, screw_pos: np.ndarray):
        """
        Execute torque balancing compensation controller for unscrewing task.
        
        Args:
            unscrewing_arm: The arm performing the unscrewing (rotating)
            anchored_arm: The arm acting as a fixed base
            screw_pos: Position of the screw (target for unscrewing)
        """
        # 1. VISUALIZE SCREW AT TARGET POSITION
        screw_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "screw")
        if screw_body_id != -1:
            self.model.body_pos[screw_body_id] = screw_pos
        
        # 2. UNSCREWING MOTION (Rotational Trajectory)
        if unscrewing_arm == 'left':
            actuator_slice = self.left_arm_actuator_slice
        else:
            actuator_slice = self.right_arm_actuator_slice
            
        unscrew_speed = 2.0  # rad/s
        dt = self.model.opt.timestep
        
        # 3. COMPENSATION CONTROL (Anchored Arm)
        # Execute one step of control
        
        # A) Move Unscrewing Arm
        # Get current joint 7 value
        current_j7_idx = actuator_slice.start + 6
        current_j7 = self.data.qpos[self.left_arm_qpos_slice.start + 6] if unscrewing_arm == 'left' else self.data.qpos[self.right_arm_qpos_slice.start + 6]
        
        # Desired velocity
        new_j7 = current_j7 + unscrew_speed * dt
        
        # Apply to control
        if unscrewing_arm == 'left':
            ctrl = self.data.qpos[self.left_arm_qpos_slice].copy()
            ctrl[6] = new_j7
            self.data.ctrl[actuator_slice] = ctrl
        else:
            ctrl = self.data.qpos[self.right_arm_qpos_slice].copy()
            ctrl[6] = new_j7
            self.data.ctrl[actuator_slice] = ctrl
            
        # B) Compensate with Anchored Arm
        body_ang_vel = self.data.qvel[3:6] # Rotational velocity of central body
        
        # Simple P-D compensation on Body Orientation (simulating arm torque)
        kp_torque = 500.0
        kd_torque = 50.0
        cancel_torque = -kd_torque * body_ang_vel
        
        if anchored_arm == 'left':
             self.maintain_anchor('left', control_anchored_arm=False)
        else:
             self.maintain_anchor('right', control_anchored_arm=False)
             
        # Add EXTRA stabilizing torque to body (representing the intelligent controller)
        self.data.xfrc_applied[self.central_body_id, 3:6] += cancel_torque
        
        return np.linalg.norm(body_ang_vel) # Return instability metric

