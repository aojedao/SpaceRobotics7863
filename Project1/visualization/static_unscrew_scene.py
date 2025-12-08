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
    
    # Position screw on back wall at goal location (moved 1m in X)
    screw_wall = 'back'
    screw_pos = np.array([1.0, 1.5, 1.5])  # Back wall position, shifted 1m in X
    
    # Find screw body
    try:
        screw_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "screw")
        model.body_pos[screw_body_id] = screw_pos
        
        # Orient screw perpendicular to back wall (pointing +Y out of wall)
        screw_quat = np.array([0.707, 0.707, 0.0, 0.0])  # 90° around X
        model.body_quat[screw_body_id] = screw_quat
        
        # Make screw visible (larger, colored)
        screw_head_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "screw_head")
        if screw_head_id != -1:
            model.geom_rgba[screw_head_id] = [1.0, 0.0, 0.0, 1.0]  # RED
            model.geom_size[screw_head_id] = [0.06, 0.02, 0]  # 6cm
        
        screw_shaft_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "screw_shaft")
        if screw_shaft_id != -1:
            model.geom_rgba[screw_shaft_id] = [1.0, 0.3, 0.0, 1.0]  # ORANGE
            model.geom_size[screw_shaft_id] = [0.025, 0.1, 0]
        
        screw_thread_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "screw_thread")
        if screw_thread_id != -1:
            model.geom_rgba[screw_thread_id] = [0.0, 1.0, 1.0, 1.0]  # CYAN
            model.geom_size[screw_thread_id] = [0.03, 0.08, 0]
        
        print(f"✓ Screw positioned at {screw_pos} on {screw_wall} wall")
        
    except Exception as e:
        print(f"⚠ Could not configure screw: {e}")
    
    # ============================================================
    # POSITION ROBOT FOR UNSCREWING POSE
    # ============================================================
    
    # Robot base positioned between screw and anchor point
    # Screw at [1.0, 1.7, 1.2], arms at body ±0.3 in X
    body_pos = np.array([0.7, 1.0, 1.2])  # Positioned to allow left arm to reach screw
    data.qpos[0:3] = body_pos
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]  # Identity quaternion
    
    # Left arm APPROACHING SCREW - aligned with screw axis
    # Left arm base at [0.7-0.3, 1.0, 1.2] = [0.4, 1.0, 1.2]
    # Screw at [1.0, 1.7, 1.2] - need to reach 0.6m in X, 0.7m in Y
    # After rotation, arm points in -X direction, so needs to bend toward screw
    left_arm_pose = np.array([
        -0.4,   # J1: Turn base toward screw (compensate for Y offset)
        0.3,    # J2: Slight shoulder lift
        0.0,    # J3: Neutral elbow rotation
        -1.1,   # J4: Moderate elbow bend to reach distance
        0.0,    # J5: Neutral wrist roll
        -0.7,   # J6: Wrist pitch to point gripper toward screw
        0.0     # J7: No rotation yet (aligned for unscrewing)
    ])
    
    # Right arm ANCHORED to back wall
    # Right arm base at [0.7+0.3, 1.0, 1.2] = [1.0, 1.0, 1.2]
    # Anchor point at [0.5, 1.7, 1.0] - reach back in X, forward in Y, down in Z
    # After rotation, arm points in +X direction
    right_arm_pose = np.array([
        0.5,    # J1: Turn base toward anchor (compensate for Y offset)
        0.5,    # J2: Lift shoulder moderately
        0.0,    # J3: Neutral elbow rotation
        -1.3,   # J4: Bend elbow to reach wall
        0.0,    # J5: Neutral wrist roll
        -0.6,   # J6: Wrist adjustment for stable grip
        0.0     # J7: Gripper closed
    ])
    
    # Set arm poses
    left_arm_start = 7
    right_arm_start = 14
    data.qpos[left_arm_start:left_arm_start+7] = left_arm_pose
    data.qpos[right_arm_start:right_arm_start+7] = right_arm_pose
    
    # Forward kinematics
    mujoco.mj_forward(model, data)
    
    # ============================================================
    # REMOVE VISUAL CLUTTER
    # ============================================================
    # Note: Waypoint spheres and markers are not in the model by default,
    # they're added by the main simulation. This scene is already clean.
    
    print("\n" + "="*60)
    print("STATIC UNSCREWING SCENE")
    print("="*60)
    print("\nScene Configuration:")
    print(f"  Screw position: {screw_pos}")
    print(f"  Robot base: {body_pos}")
    print(f"  Left arm (unscrewing): Aligned with screw")
    print(f"  Right arm (anchor): Supporting position")
    print("\nVisualization:")
    print("  - No waypoint markers")
    print("  - No highlighting spheres")
    print("  - Clean static scene for figures")
    print("\nControls:")
    print("  - Mouse: Rotate/zoom/pan view")
    print("  - Esc: Exit")
    print("="*60)
    
    # Launch viewer
    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Set camera for unscrewing view
        viewer.cam.azimuth = 45.0
        viewer.cam.elevation = -10.0
        viewer.cam.distance = 1.5
        viewer.cam.lookat[:] = screw_pos
        
        # Static visualization (no physics stepping needed)
        while viewer.is_running():
            viewer.sync()

if __name__ == "__main__":
    main()
