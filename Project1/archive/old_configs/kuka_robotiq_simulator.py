#!/usr/bin/env python3
"""
Interactive Kuka iiwa14 + Robotiq 2f85 Gripper Simulation

Controls:
- Arrow Keys: Move arm in Cartesian space
- G/H: Open/Close gripper
- R: Reset to home position
- Space: Hold to move faster
- Q: Quit

Mouse controls also available in the viewer.
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import sys
import os

class KukaGripperController:
    def __init__(self, model_path="kuka_simple_gripper.xml"):
        """Initialize the Kuka + Gripper controller"""
        
        # Check if model file exists
        if not os.path.exists(model_path):
            print(f"❌ Model file not found: {model_path}")
            print("Make sure kuka_with_robotiq_2f85.xml is in the current directory")
            sys.exit(1)
        
        try:
            print(f"Loading model: {model_path}")
            self.model = mujoco.MjModel.from_xml_path(model_path)
            self.data = mujoco.MjData(self.model)
            print("✓ Model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            print("Check that mujoco_menagerie path is correct in the XML file")
            sys.exit(1)

        # Initialize control parameters
        self.setup_control_parameters()
        
        # Movement step sizes
        self.position_step = 0.02  # meters
        self.position_step_fast = 0.05  # faster movement
        self.gripper_step = 0.01  # gripper opening step (smaller for this gripper)
        
        # Current control state
        self.target_position = np.array([0.4, 0.0, 0.3])  # Target end-effector position
        self.gripper_target = 0.0  # Robotiq 2f85 gripper (0=open, 255=closed)
        
        print("\n" + "="*60)
        print("🤖 KUKA iiwa14 + Robotiq 2f85 Interactive Simulation")
        print("="*60)
        print("Controls:")
        print("  ← → ↑ ↓  : Move arm horizontally")
        print("  Page Up/Down : Move arm up/down")
        print("  G / H     : Open / Close gripper")
        print("  R         : Reset to home position")
        print("  Space     : Hold for faster movement")
        print("  ESC / Q   : Quit simulation")
        print("  Mouse     : Click and drag in viewer")
        print("="*60)

    def setup_control_parameters(self):
        """Setup joint names and control parameters"""
        
        # Robot arm joints
        self.arm_joint_names = [
            "joint1", "joint2", "joint3", "joint4", 
            "joint5", "joint6", "joint7"
        ]
        
        # Gripper joints
        self.gripper_joint_names = [
            "left_finger_joint", "right_finger_joint"
        ]
        
        # Get joint IDs
        self.arm_joint_ids = []
        for name in self.arm_joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id >= 0:
                self.arm_joint_ids.append(joint_id)
            else:
                print(f"⚠️  Joint '{name}' not found")
        
        self.gripper_joint_ids = []
        for name in self.gripper_joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id >= 0:
                self.gripper_joint_ids.append(joint_id)
            else:
                print(f"⚠️  Joint '{name}' not found")
        
        # Get actuator IDs
        self.arm_actuator_ids = list(range(len(self.arm_joint_ids)))
        self.gripper_actuator_ids = []
        
        # Get the Robotiq 2f85 gripper actuator
        act_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator")
        if act_id >= 0:
            self.gripper_actuator_ids.append(act_id)
        
        print(f"✓ Found {len(self.arm_joint_ids)} arm joints and {len(self.gripper_joint_ids)} gripper joints")
        print(f"✓ Found {len(self.gripper_actuator_ids)} gripper actuators")

    def get_end_effector_position(self):
        """Get current end-effector position"""
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripper_center")
        if site_id >= 0:
            return self.data.site_xpos[site_id].copy()
        else:
            # Fallback to last link position
            return self.data.xpos[-1].copy()

    def inverse_kinematics_step(self, target_pos, step_size=0.05):
        """Simple Jacobian-based IK step with better stability"""
        try:
            # Get current end-effector position
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripper_center")
            if site_id < 0:
                return
            
            current_pos = self.data.site_xpos[site_id]
            
            # Position error
            pos_error = target_pos - current_pos
            error_norm = np.linalg.norm(pos_error)
            
            # If we're close enough, don't move
            if error_norm < 0.02:
                return
            
            # Limit maximum error to avoid instability
            max_error = 0.1
            if error_norm > max_error:
                pos_error = pos_error * (max_error / error_norm)
            
            # Compute Jacobian
            jacp = np.zeros((3, self.model.nv))
            jacr = np.zeros((3, self.model.nv))
            
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)
            
            # Use only position Jacobian for the arm joints
            jac = jacp[:, :len(self.arm_joint_ids)]
            
            # Check for singular configurations
            if np.linalg.det(jac @ jac.T) < 1e-6:
                return
            
            # Pseudo-inverse with adaptive damping
            damping = 0.05 + 0.1 * (1.0 / (error_norm + 0.1))
            jac_pinv = jac.T @ np.linalg.inv(jac @ jac.T + damping * np.eye(3))
            
            # Compute joint velocities with scaling
            dq = jac_pinv @ (step_size * pos_error)
            
            # Limit joint velocity changes
            max_dq = 0.1
            dq = np.clip(dq, -max_dq, max_dq)
            
            # Apply joint limits and update target smoothly
            for i, joint_id in enumerate(self.arm_joint_ids):
                if i < len(dq):
                    current_q = self.data.qpos[joint_id]
                    current_ctrl = self.data.ctrl[self.arm_actuator_ids[i]] if i < len(self.arm_actuator_ids) else current_q
                    
                    new_q = current_ctrl + 0.1 * dq[i]  # Smooth interpolation
                    
                    # Apply joint limits
                    joint_range = self.model.jnt_range[joint_id]
                    if joint_range[0] < joint_range[1]:  # Has limits
                        new_q = np.clip(new_q, joint_range[0], joint_range[1])
                    
                    # Set control target
                    if i < len(self.arm_actuator_ids):
                        self.data.ctrl[self.arm_actuator_ids[i]] = new_q
                        
        except Exception as e:
            print(f"IK error: {e}")

    def set_gripper_target(self, target):
        """Set Robotiq 2f85 gripper target (0=open, 255=closed)"""
        self.gripper_target = np.clip(target, 0.0, 255.0)
        
        # Set gripper control for Robotiq 2f85
        if len(self.gripper_actuator_ids) >= 1:
            if self.gripper_actuator_ids[0] < len(self.data.ctrl):
                self.data.ctrl[self.gripper_actuator_ids[0]] = self.gripper_target

    def reset_to_home(self):
        """Reset robot to home position"""
        self.target_position = np.array([0.4, 0.0, 0.3])
        self.gripper_target = 0.0
        
        # Reset to home keyframe if available
        home_key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if home_key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, home_key_id)
        
        print("🏠 Reset to home position")

    def handle_keyboard_input(self, key_states):
        """Handle keyboard input for robot control"""
        
        # Movement speed
        step = self.position_step_fast if key_states.get(' ', False) else self.position_step
        
        # Cartesian movement
        if key_states.get('Left', False):
            self.target_position[1] -= step  # Move left (Y-)
        if key_states.get('Right', False):
            self.target_position[1] += step  # Move right (Y+)
        if key_states.get('Up', False):
            self.target_position[0] += step  # Move forward (X+)
        if key_states.get('Down', False):
            self.target_position[0] -= step  # Move backward (X-)
        if key_states.get('Prior', False):  # Page Up
            self.target_position[2] += step  # Move up (Z+)
        if key_states.get('Next', False):   # Page Down
            self.target_position[2] -= step  # Move down (Z-)
        
        # Robotiq 2f85 gripper control (0=open, 255=closed)
        if key_states.get('g', False) or key_states.get('G', False):
            self.gripper_target = min(255.0, self.gripper_target + 25.5)  # Close gripper
        if key_states.get('h', False) or key_states.get('H', False):
            self.gripper_target = max(0.0, self.gripper_target - 25.5)    # Open gripper
        
        # Reset command
        if key_states.get('r', False) or key_states.get('R', False):
            self.reset_to_home()
        
        # Apply limits to target position
        self.target_position = np.clip(self.target_position, 
                                     [0.2, -0.5, 0.1], 
                                     [0.8, 0.5, 0.8])

    def run_simulation(self):
        """Run the interactive simulation"""
        
        # Initialize to home position
        self.reset_to_home()
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            # Store key states
            key_states = {}
            
            # Register keyboard callback
            def key_callback(keycode):
                if keycode == 256:  # ESC
                    return False
                elif keycode == ord('q') or keycode == ord('Q'):
                    return False
                
                # Map special keys
                key_map = {
                    262: 'Right', 263: 'Left', 264: 'Down', 265: 'Up',
                    266: 'Prior', 267: 'Next', 32: ' '
                }
                
                if keycode in key_map:
                    key_states[key_map[keycode]] = True
                elif 32 <= keycode <= 126:  # Printable ASCII
                    key_states[chr(keycode)] = True
                
                return True
            
            def key_release_callback(keycode):
                key_map = {
                    262: 'Right', 263: 'Left', 264: 'Down', 265: 'Up',
                    266: 'Prior', 267: 'Next', 32: ' '
                }
                
                if keycode in key_map:
                    key_states[key_map[keycode]] = False
                elif 32 <= keycode <= 126:
                    key_states[chr(keycode)] = False
            
            last_print_time = 0
            
            while viewer.is_running():
                step_start = time.time()
                
                # Handle keyboard input
                self.handle_keyboard_input(key_states)
                
                # Clear single-press keys
                for key in ['r', 'R', 'g', 'G', 'h', 'H']:
                    if key in key_states:
                        key_states[key] = False
                
                # Perform inverse kinematics
                self.inverse_kinematics_step(self.target_position, step_size=0.1)
                
                # Update gripper
                self.set_gripper_target(self.gripper_target)
                
                # Step simulation
                mujoco.mj_step(self.model, self.data)
                
                # Update viewer
                viewer.sync()
                
                # Print status occasionally
                if time.time() - last_print_time > 2.0:
                    current_pos = self.get_end_effector_position()
                    print(f"Target: [{self.target_position[0]:.2f}, {self.target_position[1]:.2f}, {self.target_position[2]:.2f}] | "
                          f"Current: [{current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f}] | "
                          f"Gripper: {self.gripper_target:.2f}")
                    last_print_time = time.time()
                
                # Timing
                elapsed = time.time() - step_start
                sleep_time = self.model.opt.timestep - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        print("👋 Simulation ended")


def main():
    """Main function"""
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = "kuka_simple_gripper.xml"
    
    try:
        controller = KukaGripperController(model_path)
        controller.run_simulation()
    except KeyboardInterrupt:
        print("\n👋 Simulation interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()
