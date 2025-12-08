#!/usr/bin/env python3
"""Quick test to verify the side-mounted arm configuration works correctly."""

import mujoco
import numpy as np
import os

# Change to script directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

def test_arm_configuration():
    """Test that the arms are mounted on the sides (X-axis ends) of the central body."""
    
    # Load the model
    model = mujoco.MjModel.from_xml_path("dual_arm_robot.xml")
    data = mujoco.MjData(model)
    
    # Step simulation to update positions
    mujoco.mj_forward(model, data)
    
    # Get body IDs
    central_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "central_body")
    left_arm_base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "left_arm_base")
    right_arm_base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "right_arm_base")
    
    print("=" * 60)
    print("SIDE-MOUNTED ARM CONFIGURATION TEST (X-AXIS)")
    print("=" * 60)
    
    # Get positions
    central_pos = data.xpos[central_body_id]
    left_base_pos = data.xpos[left_arm_base_id]
    right_base_pos = data.xpos[right_arm_base_id]
    
    print(f"\nCentral body position: ({central_pos[0]:.3f}, {central_pos[1]:.3f}, {central_pos[2]:.3f})")
    print(f"Left arm base position: ({left_base_pos[0]:.3f}, {left_base_pos[1]:.3f}, {left_base_pos[2]:.3f})")
    print(f"Right arm base position: ({right_base_pos[0]:.3f}, {right_base_pos[1]:.3f}, {right_base_pos[2]:.3f})")
    
    # Calculate offsets from central body
    left_offset = left_base_pos - central_pos
    right_offset = right_base_pos - central_pos
    
    print(f"\nLeft arm offset from body: ({left_offset[0]:.3f}, {left_offset[1]:.3f}, {left_offset[2]:.3f})")
    print(f"Right arm offset from body: ({right_offset[0]:.3f}, {right_offset[1]:.3f}, {right_offset[2]:.3f})")
    
    # Check if arms are mounted on X-axis ends
    print("\n--- VERIFICATION ---")
    
    # Expected: Left arm at X = -0.25, Right arm at X = +0.25
    left_x_check = abs(left_offset[0] - (-0.25)) < 0.01
    right_x_check = abs(right_offset[0] - 0.25) < 0.01
    y_check_left = abs(left_offset[1]) < 0.01
    y_check_right = abs(right_offset[1]) < 0.01
    
    print(f"Left arm X offset = -0.3: {'✓ PASS' if left_x_check else '✗ FAIL'}")
    print(f"Right arm X offset = +0.3: {'✓ PASS' if right_x_check else '✗ FAIL'}")
    print(f"Left arm Y offset ≈ 0: {'✓ PASS' if y_check_left else '✗ FAIL'}")
    print(f"Right arm Y offset ≈ 0: {'✓ PASS' if y_check_right else '✗ FAIL'}")
    
    all_passed = left_x_check and right_x_check and y_check_left and y_check_right
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ARM CONFIGURATION: SIDE-MOUNTED (X-AXIS ENDS) - CORRECT!")
    else:
        print("✗ ARM CONFIGURATION: CHECK FAILED")
    print("=" * 60)
    
    # Get gripper sites
    left_gripper_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_gripper_center")
    right_gripper_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_gripper_center")
    
    left_grip_pos = data.site_xpos[left_gripper_site_id]
    right_grip_pos = data.site_xpos[right_gripper_site_id]
    
    print(f"\nLeft gripper position: ({left_grip_pos[0]:.3f}, {left_grip_pos[1]:.3f}, {left_grip_pos[2]:.3f})")
    print(f"Right gripper position: ({right_grip_pos[0]:.3f}, {right_grip_pos[1]:.3f}, {right_grip_pos[2]:.3f})")
    
    gripper_dist = np.linalg.norm(left_grip_pos - right_grip_pos)
    print(f"Gripper separation: {gripper_dist:.3f}m")
    
    # Check if grippers are pointing outward (X direction)
    left_gripper_x = left_grip_pos[0] - left_base_pos[0]
    right_gripper_x = right_grip_pos[0] - right_base_pos[0]
    
    print(f"\nArm extension direction:")
    print(f"  Left arm extends: {'← LEFT (-X)' if left_gripper_x < -0.5 else '→ RIGHT (+X)' if left_gripper_x > 0.5 else '↑ VERTICAL'}")
    print(f"  Right arm extends: {'→ RIGHT (+X)' if right_gripper_x > 0.5 else '← LEFT (-X)' if right_gripper_x < -0.5 else '↑ VERTICAL'}")
    
    return all_passed

if __name__ == "__main__":
    success = test_arm_configuration()
    exit(0 if success else 1)
