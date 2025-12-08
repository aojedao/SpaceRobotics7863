"""
Static Unscrewing Scene Visualization
=====================================
Frozen scene showing the robot arm aligned with a screw.
No moving parts, no waypoint markers, clean for publication figures.
"""

import mujoco
import mujoco.viewer
import numpy as np
import os

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
MODEL_PATH = os.path.join(PROJECT_DIR, "dual_arm_robot.xml")

def main():
    """Visualize static unscrewing scene."""
    
    print(f"Loading model from: {MODEL_PATH}")
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)
    
    # Disable gravity
    model.opt.gravity[:] = [0, 0, 0]
    
    # ============================================================
    # CONFIGURE STATIC SCENE
    # ============================================================
    
    # Position screw on back wall (initial guess, will be updated to match arm)
    screw_wall = 'back'
    screw_pos = np.array([1.0, 1.5, 1.5]) 
    
    # Find screw body
    screw_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "screw")
    
    # ============================================================
    # POSITION ROBOT FOR UNSCREWING POSE
    # ============================================================
    
    # Robot base positioned (Centered in X: min=-2.1+1, max=3.9+1 -> center=1.9)
    # SHIFTED -0.5m from previous 2.4 -> Back to 1.9 (Net +1.0m from original 0.9)
    body_pos = np.array([1.9, 1.0, 1.2]) 
    data.qpos[0:3] = body_pos
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # Identity quaternion
    
    # Left arm: Reach forward to "Screw" with perpendicular orientation
    # Robot at Y=1.0. Screw area is +Y.
    # To point gripper +Y (perpendicular to wall plane), we need a 90 deg turn from X-mount.
    # J1 = 1.57 (90 deg) points +Y.
    # "aligned with the z axis" -> Make arm flatter (less shoulder lift).
    
    # Left arm: Reach forward to "Screw" with perpendicular orientation
    # OPTIMIZED ANGLES for perfect Z-axis alignment [0, 1, 0] perpendicular to wall plane
    # Computed via optimization script
    left_arm_pose = np.array([
        0.53314602, -0.13461694, 0.99939483, 
        -1.82945779, -0.21935561, -0.19491031, 
        -0.56435033
    ])
    
    # Right arm: Reach back/side to "Anchor"
    right_arm_pose = np.array([
        1.0,    # J1: Turn
        0.3,    # J2
        0.0,    # J3
        -1.5,   # J4
        0.0,    # J5
        0.0,    # J6
        0.0     # J7
    ])
    
    # Set arm poses using CORRECT INDICES
    left_arm_start = 7
    right_arm_start = 22 # Correct index for Right Arm
    
    data.qpos[left_arm_start:left_arm_start+7] = left_arm_pose
    data.qpos[right_arm_start:right_arm_start+7] = right_arm_pose
    
    # Forward kinematics to find where grippers are
    mujoco.mj_forward(model, data)
    
    # ============================================================
    # MOVE SCREW TO MATCH LEFT GRIPPER (Perfect Visual Contact)
    # ============================================================
    left_gripper_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_gripper_center")
    if left_gripper_site_id != -1:
        gripper_pos = data.site_xpos[left_gripper_site_id]
        
        # Place screw at gripper position
        screw_pos = gripper_pos.copy() 
        
        # Orientation: Align screw with gripper Z (approach axis)?
        # User requested: "invert the orientation of the z axis in the screw"
        # Previous: [0.707, 0.707, 0.0, 0.0] (+90 deg X)
        # Inverted: [0.707, -0.707, 0.0, 0.0] (-90 deg X)? Or [0.0, 0.707, 0.707, 0.0]?
        # Let's try flipping the sign of the rotation component.
        
        screw_quat = np.array([0.707, -0.707, 0.0, 0.0])  # -90 deg X?
        model.body_quat[screw_body_id] = screw_quat
        
        # Apply Offset
        # User requested: "make it move in the opposite it already did"
        # Previous: -= 0.15. New: += 0.15.
        
        screw_pos[1] += 0.02
        
        # Update screw position
        model.body_pos[screw_body_id] = screw_pos
        
        # Make screw visible
        screw_head_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "screw_head")
        if screw_head_id != -1:
            model.geom_rgba[screw_head_id] = [1.0, 0.0, 0.0, 1.0]
            model.geom_size[screw_head_id] = [0.06, 0.02, 0]
            
        screw_shaft_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "screw_shaft")
        if screw_shaft_id != -1:
            model.geom_rgba[screw_shaft_id] = [1.0, 0.3, 0.0, 1.0] 
            model.geom_size[screw_shaft_id] = [0.025, 0.1, 0]
            
        screw_thread_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "screw_thread")
        if screw_thread_id != -1:
            model.geom_rgba[screw_thread_id] = [0.0, 1.0, 1.0, 1.0]

        print(f"✓ Screw moved to match Left Gripper at {screw_pos}")
    
    print("\n" + "="*60)
    print("STATIC UNSCREWING SCENE")
    print("="*60)
    print("\nScene Configuration:")
    print(f"  Screw position: {screw_pos}")
    print(f"  Robot base: {body_pos}")
    print(f"  Left arm: Touching screw")
    print(f"  Right arm: Anchored pose (simulated)")
    print("\nControls:")
    print("  - Mouse: Rotate/zoom/pan view")
    print("  - Esc: Exit")
    print("="*60)
    
    # Launch viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Set camera
        viewer.cam.azimuth = 45.0
        viewer.cam.elevation = -10.0
        viewer.cam.distance = 1.5
        viewer.cam.lookat[:] = screw_pos
        
        # Static visualization
        while viewer.is_running():
            viewer.sync()

if __name__ == "__main__":
    main()
