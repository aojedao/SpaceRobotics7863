
import numpy as np
import time

# ============================================================================
# STUCK DETECTION AND RECOVERY MIXIN
# ============================================================================

class RecoveryMixin:
    """Methods for stuck detection and recovery strategies"""

    def check_stuck(self, current_error, current_time):
        """Monitor progress and detect stuck condition"""
        if current_error < self._stuck_best_error - 0.01:
            # Made progress
            self._stuck_best_error = current_error
            self._stuck_last_progress_time = current_time
            return False, 0.0
        
        time_since_progress = current_time - self._stuck_last_progress_time
        return time_since_progress > 2.0, time_since_progress

    def get_recovery_strategy(self, attempt_count, current_j1):
        """
        Determine the next recovery strategy based on attempt count.
        Returns: (strategy_name, target_j1_angle)
        """
        strategies = [
            # 1. Try safe neutral pose (elbow down/out)
            ("NEUTRAL", 0.0),
            
            # 2. Try rotating base slightly positive (+30 deg)
            ("+30° ROTATE", np.radians(30)),
            
            # 3. Try rotating base slightly negative (-30 deg)
            ("-30° ROTATE", np.radians(-30)),
            
            # 4. Try stronger rotation (+60 deg)
            ("+60° ROTATE", np.radians(60)),
            
            # 5. Try stronger rotation (-60 deg)
            ("-60° ROTATE", np.radians(-60)),
            
            # 6. Try extreme rotation (+90 deg)
            ("+90° ROTATE", np.radians(90)),
            
            # 7. Try extreme rotation (-90 deg)
            ("-90° ROTATE", np.radians(-90)),
            
            # 8. Try flip (if joint limits allow, ±135 deg)
            ("+135° ROTATE", np.radians(135)),
            ("-135° ROTATE", np.radians(-135)),
            
            # 9. Panic/Full extension attempt (neutral)
            ("FULL RESET", 0.0)
        ]
        
        # Cycle through strategies if attempts > available strategies
        idx = (attempt_count - 1) % len(strategies)
        return strategies[idx]

    def execute_recovery_step(self, moving_arm, target_j1, dt):
        """Execute one step of the recovery movement"""
        if moving_arm == 'left':
            qpos_slice = self.left_arm_qpos_slice
            actuator_slice = self.left_arm_actuator_slice
        else:
            qpos_slice = self.right_arm_qpos_slice
            actuator_slice = self.right_arm_actuator_slice
        
        # Smoothly rotate joint1 towards target
        current_j1 = self.data.qpos[qpos_slice][0]
        j1_error = target_j1 - current_j1
        
        # Wrap error
        while j1_error > np.pi: j1_error -= 2*np.pi
        while j1_error < -np.pi: j1_error += 2*np.pi
        
        # Only control Joint 1, keep others at current positions
        current_q = self.data.qpos[qpos_slice].copy()
        
        # Proportional control for recovery
        recover_speed = 2.0 # rad/s
        delta = np.clip(j1_error, -recover_speed * dt, recover_speed * dt)
        
        current_q[0] += delta
        self.data.ctrl[actuator_slice] = current_q
        
        return abs(j1_error) < 0.1 # Return True if reached target

    def execute_advanced_recovery_step(self, moving_arm: str, recovery_joint1_target: float, 
                                     recovery_j1_velocity_gain: float, dt: float):
        """
        Execute advanced recovery maneuver:
        1. Rotate Joint 1 towards target
        2. Retract shoulder (J2) and bend elbow (J4) ("Turtle" pose)
        3. Apply coordinated control to input arm AND anchored arm (Dual-Arm Recovery)
        """
        if moving_arm == 'left':
            qpos_slice = self.left_arm_qpos_slice
            actuator_slice = self.left_arm_actuator_slice
            anchored_actuator_slice = self.right_arm_actuator_slice
            anchored_qpos_slice = self.right_arm_qpos_slice
        else:
            qpos_slice = self.right_arm_qpos_slice
            actuator_slice = self.right_arm_actuator_slice
            anchored_actuator_slice = self.left_arm_actuator_slice
            anchored_qpos_slice = self.left_arm_qpos_slice
        
        # Smoothly rotate joint1 towards target
        current_j1 = self.data.qpos[qpos_slice][0]
        j1_error = recovery_joint1_target - current_j1
        
        # Normalize angle error to [-pi, pi]
        while j1_error > np.pi:
            j1_error -= 2 * np.pi
        while j1_error < -np.pi:
            j1_error += 2 * np.pi
        
        # Apply joint1 rotation with optimized gain
        j1_velocity = recovery_j1_velocity_gain * j1_error
        new_j1 = current_j1 + j1_velocity * dt
        
        # SMART RECOVERY: Also retract shoulder (J2) and bend elbow (J4) to "Turtle" the arm
        target_j2 = -1.0
        target_j4 = 1.5
        target_j3 = 0.0
        
        current_j2 = self.data.qpos[qpos_slice][1]
        current_j4 = self.data.qpos[qpos_slice][3]
        
        new_j2 = current_j2 + 2.0 * (target_j2 - current_j2) * dt
        new_j4 = current_j4 + 2.0 * (target_j4 - current_j4) * dt
        
        # Set commands for Moving Arm
        current_cmd = self.data.ctrl[actuator_slice].copy()
        current_cmd[0] = new_j1  # Rotate
        current_cmd[1] = new_j2  # Retract Shoulder
        current_cmd[3] = new_j4  # Bend Elbow
        self.data.ctrl[actuator_slice] = current_cmd
        
        # Anchored Arm Control: Rotate J1, Retract J2/J4
        anchored_current_q = self.data.qpos[anchored_qpos_slice]
        anchored_current_j1 = anchored_current_q[0]
        anchored_current_j2 = anchored_current_q[1]
        anchored_current_j3 = anchored_current_q[2]
        anchored_current_j4 = anchored_current_q[3]
        
        # Calculate anchored errors
        anchored_j1_error = recovery_joint1_target - anchored_current_j1
        while anchored_j1_error > np.pi: anchored_j1_error -= 2*np.pi
        while anchored_j1_error < -np.pi: anchored_j1_error += 2*np.pi
        
        anchored_cmd = self.data.ctrl[anchored_actuator_slice].copy()
        # Apply P-control to anchored arm too
        anchored_cmd[0] += 2.0 * anchored_j1_error * dt
        anchored_cmd[1] += 2.0 * (target_j2 - anchored_current_j2) * dt
        anchored_cmd[2] += 2.0 * (target_j3 - anchored_current_j3) * dt
        anchored_cmd[3] += 2.0 * (target_j4 - anchored_current_j4) * dt
        
        self.data.ctrl[anchored_actuator_slice] = anchored_cmd
