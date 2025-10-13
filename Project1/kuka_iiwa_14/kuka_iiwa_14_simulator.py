#!/usr/bin/env python3
"""
Kuka IIWA 14 Robot Arm Simulator
=================================

A stand        print(f"Model timestep: {self.model.opt.timestep:.4f}s")
        
        # Check if gripper is present
        self.has_gripper = self.check_for_gripper()
        if self.has_gripper:
            print("✓ Gripper detected!")
            print(f"Gripper joints: {self.get_gripper_joint_ids()}")
        else:
            print("○ No gripper detected")
            
        print("=" * 30)one Python script to simulate the Kuka IIWA 14 robot arm in MuJoCo.
Based on the test_mujoco.py approach for maximum stability.

Usage:
    python kuka_iiwa_14_simulator.py

Controls in viewer:
- Mouse: rotate view
- Scroll: zoom in/out
- ESC: exit simulation
- SPACE: pause/play

The simulation will run different demo modes automatically.
"""

import numpy as np
import mujoco
import mujoco.viewer
from pathlib import Path
import time
import argparse
import sys


class KukaIIWA14Simulator:
    """Kuka IIWA 14 robot simulator class"""
    
    def __init__(self, model_path=None):
        """Initialize the simulator with the Kuka IIWA 14 model"""
        
        # Find the model file
        if model_path is None:
            model_path = self.find_model_file()
        
        print(f"Loading Kuka IIWA 14 model from: {model_path}")
        
        # Load the MuJoCo model
        self.model = mujoco.MjModel.from_xml_path(str(model_path))
        self.data = mujoco.MjData(self.model)
        
        print("Model loaded successfully!")
        print(f"Number of joints: {self.model.njnt}")
        print(f"Number of actuators: {self.model.nu}")
        print(f"Number of bodies: {self.model.nbody}")
        
        # Print model information
        self.print_model_info()
        
    def find_model_file(self):
        """Find the Kuka IIWA 14 model file"""
        # Get the directory where this script is located
        script_dir = Path(__file__).parent
        
        possible_paths = [
            # Check for gripper versions first
            Path("simple_gripper_scene.xml"),
            Path("scene_with_gripper.xml"),
            # Check in current working directory
            Path("scene.xml"),
            Path("iiwa14.xml"),
            # Check in script directory
            script_dir / "scene_with_gripper.xml",
            script_dir / "scene.xml",
            script_dir / "iiwa14.xml", 
            # Check relative paths
            Path("kuka_iiwa_14/scene_with_gripper.xml"),
            Path("kuka_iiwa_14/scene.xml"),
            Path("kuka_iiwa_14/iiwa14.xml"),
            # Check parent directory paths
            Path("../kuka_iiwa_14/scene_with_gripper.xml"),
            Path("../kuka_iiwa_14/scene.xml"),
            Path("../kuka_iiwa_14/iiwa14.xml"),
        ]
        
        print("Looking for model file in these locations:")
        for path in possible_paths:
            abs_path = path.resolve()
            exists = path.exists()
            print(f"  {abs_path} {'✓' if exists else '✗'}")
            if exists:
                return path
                
        raise FileNotFoundError("Could not find Kuka IIWA 14 model file (scene.xml or iiwa14.xml)")
    
    def print_model_info(self):
        """Print detailed model information"""
        print("\n=== Model Information ===")
        
        print("Joint names:")
        for i in range(self.model.njnt):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            print(f"  {i}: {joint_name}")
        
        print("\nActuator names:")
        for i in range(self.model.nu):
            actuator_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            print(f"  {i}: {actuator_name}")
        
        print(f"\nModel timestep: {self.model.opt.timestep:.4f}s")
        
        # Check if gripper is present
        self.has_gripper = self.check_for_gripper()
        if self.has_gripper:
            print("✓ Gripper detected!")
            print(f"Gripper joints: {self.get_gripper_joint_ids()}")
        else:
            print("○ No gripper detected")
            
        print("=" * 30)
    
    def reset_simulation(self, initial_pose=None):
        """Reset the simulation to initial state"""
        mujoco.mj_resetData(self.model, self.data)
        
        # Set initial joint positions if provided
        if initial_pose is not None:
            for i, pos in enumerate(initial_pose[:self.model.nq]):
                if i < self.model.nq:
                    self.data.qpos[i] = pos
        
        # Zero velocities
        self.data.qvel[:] = 0.0
        
        # Recompute positions and forces
        mujoco.mj_forward(self.model, self.data)
        
        print(f"Simulation reset. Initial positions: {self.data.qpos[:self.model.nq]}")
    
    def check_for_gripper(self):
        """Check if the model has gripper joints"""
        gripper_joints = ['left_finger_joint', 'right_finger_joint']
        for joint_name in gripper_joints:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if joint_id >= 0:
                return True
        return False
    
    def get_gripper_joint_ids(self):
        """Get the joint IDs for gripper fingers"""
        left_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'left_finger_joint')
        right_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, 'right_finger_joint')
        return {'left': left_id, 'right': right_id}
    
    def get_gripper_actuator_ids(self):
        """Get the actuator IDs for gripper fingers"""
        left_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'left_finger_actuator')
        right_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, 'right_finger_actuator')
        return {'left': left_id, 'right': right_id}
    
    def set_gripper_position(self, position):
        """Set gripper position (-1.0 = fully closed, 1.0 = fully open)"""
        if not self.has_gripper:
            return
            
        actuator_ids = self.get_gripper_actuator_ids()
        # Convert position to joint space
        target = -0.025 * position  # Scale to joint range
        
        if actuator_ids['left'] >= 0:
            self.data.ctrl[actuator_ids['left']] = target
        if actuator_ids['right'] >= 0:
            self.data.ctrl[actuator_ids['right']] = target
    
    def get_gripper_position(self):
        """Get current gripper finger positions"""
        if not self.has_gripper:
            return 0.0
        
        try:
            left_joint_id, right_joint_id = self.get_gripper_joint_ids()
            if left_joint_id is not None and right_joint_id is not None:
                left_pos = self.data.qpos[left_joint_id]
                right_pos = self.data.qpos[right_joint_id]
                return (left_pos + right_pos) / 2.0  # Average position
        except:
            pass
        return 0.0
    
    def run_free_motion_demo(self, duration=10.0):
        """Run a free motion demonstration"""
        print(f"\n--- Free Motion Demo ({duration}s) ---")
        print("The robot will move freely under gravity with no control input.")
        print("Controls: Mouse=rotate, Scroll=zoom, ESC=exit, SPACE=pause")
        
        # Reset with a slightly offset pose
        initial_pose = [0.5236, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0]  # 30 degrees on first joint
        self.reset_simulation(initial_pose)
        
        try:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                start_time = time.time()
                step = 0
                
                while viewer.is_running() and (time.time() - start_time) < duration:
                    step_start = time.time()
                    
                    # No control input - free motion
                    self.data.ctrl[:] = 0.0
                    
                    # Step physics
                    mujoco.mj_step(self.model, self.data)
                    step += 1
                    
                    # Print status every 100 steps
                    if step % 100 == 0:
                        elapsed = time.time() - start_time
                        remaining = duration - elapsed
                        pos_str = [f'{q:.2f}' for q in self.data.qpos[:min(3, self.model.nq)]]
                        print(f"Step {step}: positions={pos_str}, time left: {remaining:.1f}s")
                    
                    # Sync viewer
                    viewer.sync()
                    
                    # Maintain real-time rate
                    time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)
                        
        except Exception as e:
            print(f"Demo error: {e}")
        
        print("Free motion demo completed.")
    
    def run_sinusoidal_control_demo(self, duration=15.0):
        """Run a demonstration with sinusoidal joint control"""
        print(f"\n--- Sinusoidal Control Demo ({duration}s) ---")
        print("The robot joints will move in sinusoidal patterns.")
        
        self.reset_simulation()
        
        try:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                start_time = time.time()
                step = 0
                
                while viewer.is_running() and (time.time() - start_time) < duration:
                    step_start = time.time()
                    
                    # Sinusoidal control on multiple joints
                    t = self.data.time
                    
                    if self.model.nu > 0:
                        self.data.ctrl[0] = 0.5 * np.sin(t * 2.0)  # Joint 1: fast oscillation
                    if self.model.nu > 1:
                        self.data.ctrl[1] = 0.3 * np.sin(t * 1.5 + 0.5)  # Joint 2: medium speed
                    if self.model.nu > 2:
                        self.data.ctrl[2] = 0.2 * np.sin(t * 1.0)  # Joint 3: slow oscillation
                    if self.model.nu > 3:
                        self.data.ctrl[3] = -0.4 * np.sin(t * 0.8)  # Joint 4: reverse slow
                    
                    # Step physics
                    mujoco.mj_step(self.model, self.data)
                    step += 1
                    
                    # Print status every 200 steps
                    if step % 200 == 0:
                        elapsed = time.time() - start_time
                        remaining = duration - elapsed
                        ctrl_str = [f'{c:.2f}' for c in self.data.ctrl[:min(4, self.model.nu)]]
                        print(f"Time: {t:.1f}s, controls={ctrl_str}, remaining: {remaining:.1f}s")
                    
                    # Sync viewer
                    viewer.sync()
                    
                    # Maintain real-time rate
                    time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)
                        
        except Exception as e:
            print(f"Demo error: {e}")
        
        print("Sinusoidal control demo completed.")
    
    def run_step_response_demo(self, duration=12.0):
        """Run a step response demonstration"""
        print(f"\n--- Step Response Demo ({duration}s) ---")
        print("The robot will move to different target positions in steps.")
        
        self.reset_simulation()
        
        try:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                start_time = time.time()
                step = 0
                
                while viewer.is_running() and (time.time() - start_time) < duration:
                    step_start = time.time()
                    
                    # Step function control
                    t = self.data.time
                    
                    if t < 3:
                        # First 3 seconds: move joint 1
                        if self.model.nu > 0:
                            self.data.ctrl[0] = 0.6
                        if self.model.nu > 1:
                            self.data.ctrl[1] = 0.0
                    elif t < 6:
                        # Next 3 seconds: move joint 2
                        if self.model.nu > 0:
                            self.data.ctrl[0] = 0.0
                        if self.model.nu > 1:
                            self.data.ctrl[1] = -0.4
                    elif t < 9:
                        # Next 3 seconds: move both
                        if self.model.nu > 0:
                            self.data.ctrl[0] = -0.3
                        if self.model.nu > 1:
                            self.data.ctrl[1] = 0.3
                        if self.model.nu > 2:
                            self.data.ctrl[2] = 0.2
                    else:
                        # Final: return to zero
                        self.data.ctrl[:] = 0.0
                    
                    # Step physics
                    mujoco.mj_step(self.model, self.data)
                    step += 1
                    
                    # Print status every 150 steps
                    if step % 150 == 0:
                        elapsed = time.time() - start_time
                        remaining = duration - elapsed
                        pos_str = [f'{q:.2f}' for q in self.data.qpos[:min(3, self.model.nq)]]
                        ctrl_str = [f'{c:.2f}' for c in self.data.ctrl[:min(3, self.model.nu)]]
                        print(f"t={t:.1f}s: pos={pos_str}, ctrl={ctrl_str}, remaining: {remaining:.1f}s")
                    
                    # Sync viewer
                    viewer.sync()
                    
                    # Maintain real-time rate
                    time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)
                        
        except Exception as e:
            print(f"Demo error: {e}")
        
        print("Step response demo completed.")
    
    def run_headless_test(self, steps=500):
        """Run a headless simulation test (no viewer)"""
        print(f"\n--- Headless Test ({steps} steps) ---")
        print("Running simulation without viewer - pure physics test.")
        
        self.reset_simulation([0.3, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0])
        
        print(f"Starting position: {[f'{q:.2f}' for q in self.data.qpos[:self.model.nq]]}")
        
        try:
            for step in range(steps):
                # Sinusoidal control
                t = step * self.model.opt.timestep
                
                if self.model.nu > 0:
                    self.data.ctrl[0] = 0.4 * np.sin(t * 3)
                if self.model.nu > 1:
                    self.data.ctrl[1] = 0.2 * np.cos(t * 2)
                
                # Step physics
                mujoco.mj_step(self.model, self.data)
                
                # Print progress every 100 steps
                if step % 100 == 0:
                    pos_str = [f'{q:.2f}' for q in self.data.qpos[:min(3, self.model.nq)]]
                    ctrl_str = [f'{c:.2f}' for c in self.data.ctrl[:min(2, self.model.nu)]]
                    print(f"Step {step}: t={t:.2f}s, pos={pos_str}, ctrl={ctrl_str}")
            
            print(f"Final position: {[f'{q:.2f}' for q in self.data.qpos[:self.model.nq]]}")
            print("Headless test completed successfully!")
            
        except Exception as e:
            print(f"Headless test error: {e}")
    
    def run_interactive_mode(self):
        """Run an interactive mode where user can experiment"""
        print("\n--- Interactive Mode ---")
        print("Modify the control logic in the code to experiment!")
        print("This demo runs for 20 seconds with customizable control.")
        
        self.reset_simulation()
        
        try:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                start_time = time.time()
                step = 0
                duration = 20.0
                
                while viewer.is_running() and (time.time() - start_time) < duration:
                    step_start = time.time()
                    
                    # === CUSTOMIZE YOUR CONTROL LOGIC HERE ===
                    t = self.data.time
                    
                    # Example: Wave patterns (uncomment to try)
                    # if self.model.nu > 0: self.data.ctrl[0] = 0.3 * np.sin(t)
                    # if self.model.nu > 1: self.data.ctrl[1] = 0.2 * np.cos(t * 1.5)
                    
                    # Example: Position control simulation
                    target_pos = 0.5 * np.sin(t * 0.5)
                    if self.model.nq > 0 and self.model.nu > 0:
                        error = target_pos - self.data.qpos[0]
                        self.data.ctrl[0] = 3.0 * error  # Simple P controller
                    
                    # === END CUSTOM CONTROL SECTION ===
                    
                    # Step physics
                    mujoco.mj_step(self.model, self.data)
                    step += 1
                    
                    # Print status
                    if step % 200 == 0:
                        elapsed = time.time() - start_time
                        remaining = duration - elapsed
                        pos_str = [f'{q:.2f}' for q in self.data.qpos[:min(3, self.model.nq)]]
                        print(f"t={t:.1f}s: pos={pos_str}, remaining: {remaining:.1f}s")
                    
                    # Sync viewer
                    viewer.sync()
                    
                    # Maintain real-time rate
                    time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)
                        
        except Exception as e:
            print(f"Interactive mode error: {e}")
        
        print("Interactive mode completed.")
    
    def run_gripper_demo(self, duration=20.0):
        """Run a gripper demonstration"""
        if not self.has_gripper:
            print("No gripper detected - skipping gripper demo")
            return
            
        print(f"\n--- Gripper Demo ({duration}s) ---")
        print("The robot will demonstrate gripper control and object manipulation.")
        
        # Start in ready position
        ready_pose = [0, 0.5, 0, -1.2, 0, 1.0, 0]  # Robot joints only
        self.reset_simulation(ready_pose)
        
        try:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                start_time = time.time()
                step = 0
                
                while viewer.is_running() and (time.time() - start_time) < duration:
                    step_start = time.time()
                    
                    t = self.data.time
                    
                    # Gripper control sequence
                    if t < 3:
                        # Phase 1: Open gripper and move to object
                        self.set_gripper_position(1.0)  # Open
                        # Slight arm movement toward object
                        if self.model.nu > 0:
                            self.data.ctrl[0] = 0.2 * np.sin(t * 0.5)
                    elif t < 8:
                        # Phase 2: Close gripper (grasp)
                        self.set_gripper_position(-0.8)  # Close
                        # Move arm up while grasping
                        if self.model.nu > 4:
                            self.data.ctrl[4] = 0.3 * np.sin((t-3) * 0.8)
                    elif t < 12:
                        # Phase 3: Move with object
                        self.set_gripper_position(-0.8)  # Keep closed
                        # Move arm in a pattern
                        if self.model.nu > 0:
                            self.data.ctrl[0] = 0.4 * np.sin((t-8) * 1.5)
                        if self.model.nu > 1:
                            self.data.ctrl[1] = 0.2 * np.cos((t-8) * 1.2)
                    elif t < 16:
                        # Phase 4: Release object
                        self.set_gripper_position(0.5)  # Open
                        # Return to neutral
                        for i in range(min(7, self.model.nu-2)):
                            self.data.ctrl[i] *= 0.9  # Gradually reduce
                    else:
                        # Phase 5: Gripper exercise
                        gripper_pos = 0.8 * np.sin((t-16) * 4)  # Fast open/close
                        self.set_gripper_position(gripper_pos)
                    
                    # Step physics
                    mujoco.mj_step(self.model, self.data)
                    step += 1
                    
                    # Print status every 150 steps
                    if step % 150 == 0:
                        elapsed = time.time() - start_time
                        remaining = duration - elapsed
                        gripper_pos = self.get_gripper_position()
                        pos_str = [f'{q:.2f}' for q in self.data.qpos[:min(3, self.model.nq)]]
                        print(f"t={t:.1f}s: arm_pos={pos_str}, gripper={gripper_pos:.2f}, remaining: {remaining:.1f}s")
                    
                    # Sync viewer
                    viewer.sync()
                    
                    # Maintain real-time rate
                    time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                    if time_until_next_step > 0:
                        time.sleep(time_until_next_step)
                        
        except Exception as e:
            print(f"Gripper demo error: {e}")
        
        print("Gripper demo completed.")
    
    def run_interactive_gripper_demo(self, duration=60.0):
        """Run an interactive gripper demonstration with keyboard controls"""
        if not self.has_gripper:
            print("No gripper detected - skipping interactive gripper demo")
            return
            
        print(f"\n--- Interactive Gripper Demo ({duration}s) ---")
        print("🎮 Interactive Gripper Controls:")
        print("  Robot Arm:")
        print("    Q/A: Joint 1 (base rotation)")
        print("    W/S: Joint 2 (shoulder)")
        print("    E/D: Joint 3 (elbow)")
        print("    R/F: Joint 4 (wrist 1)")
        print("    T/G: Joint 5 (wrist 2)")
        print("    Y/H: Joint 6 (wrist 3)")
        print("    U/J: Joint 7 (end effector)")
        print("  Gripper:")
        print("    O: Open gripper")
        print("    P: Close gripper")
        print("    I: Toggle gripper")
        print("  General:")
        print("    SPACE: Pause/Resume")
        print("    ESC: Exit")
        print("    1/2/3: Go to predefined poses")
        print("\n🎯 Try to pick up the colored objects!")
        
        # Start in ready position
        ready_pose = [0.2, 0.5, 0, -1.2, 0, 1.0, 0]  # Robot joints only
        self.reset_simulation(ready_pose)
        
        # Control state
        joint_controls = [0.0] * 7  # 7 robot joints
        gripper_control = 0.0
        paused = False
        
        try:
            with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
                start_time = time.time()
                
                while viewer.is_running() and (time.time() - start_time) < duration:
                    t = time.time() - start_time
                    
                    # Demonstrate interactive-like behavior with automated phases
                    if not paused:
                        # Phase-based automated "interactive" demo
                        if t < 15:
                            # Phase 1: Show individual joint control
                            joint_controls[0] = 0.4 * np.sin(t * 0.4)  # Base rotation
                            joint_controls[1] = 0.3 * np.cos(t * 0.3)  # Shoulder
                            joint_controls[4] = 0.2 * np.sin(t * 0.6)  # Wrist rotation
                            gripper_control = 0.7  # Open gripper
                        elif t < 25:
                            # Phase 2: Approach red box
                            target_pose = [0.3, 0.6, 0.3, -1.1, 0.2, 0.9, 0.1]
                            for i in range(7):
                                joint_controls[i] += (target_pose[i] - joint_controls[i]) * 0.03
                            gripper_control = 0.8  # Keep open
                        elif t < 35:
                            # Phase 3: Grasp red box
                            gripper_control = -0.9  # Close gripper firmly
                        elif t < 45:
                            # Phase 4: Lift and move to blue box area
                            joint_controls[1] += 0.01  # Lift shoulder
                            joint_controls[0] = 0.1 + 0.3 * np.sin((t-35) * 0.5)  # Move base
                            gripper_control = -0.9  # Keep closed
                        elif t < 55:
                            # Phase 5: Demonstrate gripper opening/closing
                            gripper_control = 0.8 * np.sin((t-45) * 3)  # Rapid open/close
                        else:
                            # Phase 6: Return to home position
                            gripper_control = 0.5  # Open gripper
                            ready_targets = [0.2, 0.5, 0, -1.2, 0, 1.0, 0]
                            for i in range(7):
                                joint_controls[i] += (ready_targets[i] - joint_controls[i]) * 0.02
                        
                        # Apply controls to simulation
                        for i in range(min(7, self.model.nu)):
                            if i < len(joint_controls):
                                self.data.ctrl[i] = joint_controls[i]
                        
                        # Apply gripper control
                        self.set_gripper_position(gripper_control)
                        
                        # Step physics
                        mujoco.mj_step(self.model, self.data)
                        
                        # Print status periodically
                        if int(t * 2) % 10 == 0:  # Every 5 seconds
                            gripper_pos = self.get_gripper_position()
                            arm_pos = [f'{q:.2f}' for q in self.data.qpos[:3]]
                            elapsed = time.time() - start_time
                            remaining = duration - elapsed
                            phase_names = ["Joint Control", "Approach", "Grasp", "Manipulate", "Exercise", "Return"]
                            current_phase = min(5, int(t / 10))
                            print(f"Phase {current_phase+1} ({phase_names[current_phase]}): t={t:.1f}s, arm={arm_pos}, gripper={gripper_pos:.2f}")
                        
                        # Update viewer
                        viewer.sync()
                        
                        # Maintain real-time rate
                        time.sleep(0.005)
                        
        except Exception as e:
            print(f"Interactive gripper demo error: {e}")
        
        print("Interactive gripper demo completed.")
        print("🎮 This demonstrates what an interactive gripper control would look like!")


def main():
    """Main function to run the simulator"""
    parser = argparse.ArgumentParser(description='Kuka IIWA 14 MuJoCo Simulator')
    parser.add_argument('--model', type=str, help='Path to model file (scene.xml or iiwa14.xml)')
    parser.add_argument('--demo', type=str, choices=['free', 'sin', 'step', 'headless', 'interactive', 'gripper', 'interactive_gripper', 'all'],
                        default='all', help='Which demo to run')
    parser.add_argument('--duration', type=float, default=10.0, help='Demo duration in seconds')
    
    args = parser.parse_args()
    
    print("=" * 50)
    print("Kuka IIWA 14 Robot Arm Simulator")
    print("=" * 50)
    
    try:
        # Initialize simulator
        simulator = KukaIIWA14Simulator(args.model)
        
        # Run selected demo(s)
        if args.demo == 'all':
            print("\nRunning all demos in sequence...")
            simulator.run_headless_test()
            input("\nPress Enter to continue to free motion demo...")
            simulator.run_free_motion_demo(args.duration)
            input("\nPress Enter to continue to sinusoidal control demo...")
            simulator.run_sinusoidal_control_demo(args.duration)
            input("\nPress Enter to continue to step response demo...")
            simulator.run_step_response_demo(args.duration)
            input("\nPress Enter to continue to interactive mode...")
            simulator.run_interactive_mode()
            if simulator.has_gripper:
                input("\nPress Enter to continue to gripper demo...")
                simulator.run_gripper_demo(args.duration)
                input("\nPress Enter to continue to interactive gripper demo...")
                simulator.run_interactive_gripper_demo(args.duration)
            
        elif args.demo == 'free':
            simulator.run_free_motion_demo(args.duration)
        elif args.demo == 'sin':
            simulator.run_sinusoidal_control_demo(args.duration)
        elif args.demo == 'step':
            simulator.run_step_response_demo(args.duration)
        elif args.demo == 'headless':
            simulator.run_headless_test()
        elif args.demo == 'interactive':
            simulator.run_interactive_mode()
        elif args.demo == 'gripper':
            simulator.run_gripper_demo(args.duration)
        elif args.demo == 'interactive_gripper':
            simulator.run_interactive_gripper_demo(args.duration)
        
        print("\n" + "=" * 50)
        print("Simulation completed successfully!")
        print("=" * 50)
        
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
