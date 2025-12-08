"""
Simple Bimanual Robot Visualization for Static Photos
=====================================================
This script spawns only the dual-arm robot (no environment) 
for clean visualization and photo capture.
"""

import mujoco
import mujoco.viewer
import numpy as np
import os

# Get the path to the robot model
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
MODEL_PATH = os.path.join(PROJECT_DIR, "dual_arm_robot.xml")

def main():
    """Load and visualize the bimanual robot."""
    
    # Load the MuJoCo model
    print(f"Loading model from: {MODEL_PATH}")
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    
    # Disable gravity for clean visualization
    model.opt.gravity[:] = [0, 0, 0]
    
    # Set robot to a neutral, photogenic pose with mid-range joints
    # All joints at 0 (middle of range: J1,J3,J5,J7: ±2.97 rad, J2,J4,J6: ±2.09 rad)
    neutral_pose = np.zeros(7)
    
    # Left arm - neutral pose
    left_arm_pose = neutral_pose.copy()
    
    # Right arm - neutral pose
    right_arm_pose = neutral_pose.copy()
    
    # Find arm joint indices
    left_arm_start = 7  # After base (7 DOF: 3 pos + 4 quat)
    right_arm_start = 14  # After base + left arm
    
    # Set arm poses
    data.qpos[left_arm_start:left_arm_start+7] = left_arm_pose
    data.qpos[right_arm_start:right_arm_start+7] = right_arm_pose
    
    # Set base position inside ISS box (center of workspace)
    # ISS module bounds: x=[-2.1, 3.9], y=[-0.5, 1.7], z=[0.1, 2.2]
    data.qpos[0:3] = [0.9, 0.6, 1.15]  # Center position
    # Quaternion for 90° Y rotation: [cos(45°), 0, sin(45°), 0] = [0.707, 0, 0.707, 0]
    data.qpos[3:7] = [0.707, 0.0, 0.707, 0.0]
    
    # Forward kinematics to compute positions
    mujoco.mj_forward(model, data)
    
    print("\n" + "="*60)
    print("BIMANUAL ROBOT VISUALIZATION")
    print("="*60)
    print("\nControls:")
    print("  - Mouse drag: Rotate view")
    print("  - Scroll: Zoom in/out")
    print("  - Right-click drag: Pan")
    print("  - Space: Pause/Resume")
    print("  - Esc: Exit")
    print("\nRobot Configuration:")
    print(f"  Base position: {data.qpos[0:3]}")
    print(f"  Left arm joints: {left_arm_pose}")
    print(f"  Right arm joints: {right_arm_pose}")
    print("\nTIP: Adjust camera angle and take screenshots for report figures")
    print("="*60)
    
    # Launch passive viewer (no control, just visualization)
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Set camera for good viewing angle
        viewer.cam.azimuth = 90.0
        viewer.cam.elevation = -20.0
        viewer.cam.distance = 2.5
        viewer.cam.lookat[:] = [0.0, 0.0, 0.0]
        
        # Run visualization loop
        while viewer.is_running():
            # Just sync viewer - NO physics step to keep robot static
            viewer.sync()

if __name__ == "__main__":
    main()
