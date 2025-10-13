#!/usr/bin/env python3
"""
Integrated Zero-Gravity Simulation: Original KUKA iiwa14 + Robotiq Gripper + Box/Door Model

This simulation uses the original iiwa14.xml as the base and adds the gripper and door system on top.
This ensures we get the exact collision and physics properties from the original KUKA model.

Mouse controls are available in the viewer for camera movement.
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import argparse
import matplotlib.pyplot as plt
import os
from scipy.spatial.transform import Rotation
from collections import deque

class IntegratedZeroGravitySimulation:
    def __init__(self, 
                 iiwa_model_path="kuka_iiwa_14/iiwa14.xml",
                 door_model_path="door_hinge_model.xml"):
        """Initialize the integrated simulation using original iiwa14 + extensions"""
        
        # Check if model files exist
        if not os.path.exists(iiwa_model_path):
            raise FileNotFoundError(f"KUKA iiwa14 model not found: {iiwa_model_path}")
            
        if not os.path.exists(door_model_path):
            raise FileNotFoundError(f"Door model not found: {door_model_path}")

        # Create integrated model by extending iiwa14.xml
        self.create_integrated_model(iiwa_model_path, door_model_path)
        
        # Initialize control parameters
        self.setup_control_parameters()
        
        # Movement step sizes
        self.position_step = 0.02  # meters
        self.position_step_fast = 0.05  # faster movement
        self.impulse_strength = 2.0  # impulse strength for box movement
        
        # Controller parameters (tune these for your controller)
        self.kp_position = 100.0    # Proportional gain for position control
        self.kd_position = 20.0     # Derivative gain for position control
        self.max_joint_velocity = 2.0  # Maximum allowed velocity per joint (rad/s)
        
        # Controller state variables
        self.previous_position_error = np.zeros(3)  # For derivative control
        self.controller_enabled = True  # Enable/disable automatic controller (default: ON)
        
        # Other control state
        self.gripper_target = 0.0  # Robotiq 2f85 gripper (0=open, 255=closed)
        
        # Data collection for plotting
        self.max_data_points = 100000  # Maximum number of data points to store
        self.time_data = deque(maxlen=self.max_data_points)
        self.position_error_data = deque(maxlen=self.max_data_points)  # 3D position error components
        self.target_position_data = deque(maxlen=self.max_data_points)  # 3D target position components
        self.orientation_error_data = deque(maxlen=self.max_data_points)  # 3D orientation error components
        self.current_position_data = deque(maxlen=self.max_data_points)
        self.current_orientation_data = deque(maxlen=self.max_data_points)
        self.target_orientation_data = deque(maxlen=self.max_data_points)  # Target orientation
        self.joint_values_data = deque(maxlen=self.max_data_points)
        self.joint_velocities_data = deque(maxlen=self.max_data_points)  # Joint velocities
        self.linear_velocity_data = deque(maxlen=self.max_data_points)   # End effector linear velocity
        self.angular_velocity_data = deque(maxlen=self.max_data_points)  # End effector angular velocity
        self.simulation_start_time = None
        
        print("\n" + "="*70)
        print("🚀 INTEGRATED ZERO-GRAVITY PHYSICS SIMULATION")
        print("🤖 Original KUKA iiwa14 + Robotiq 2F85 Gripper + Box/Door Model")
        print("="*70)
        print("INTEGRATED CONTROLLER FRAMEWORK:")
        print("  • Original iiwa14.xml collision and physics properties")
        print("  • 6DOF Cartesian space control with quaternion orientation")
        print("  • Redundant manipulator control with null space projection")
        print("  • Damped least squares to avoid singularities")
        print("  • Angular velocity damping for orientation stability")
        print("  • Comprehensive real-time data collection and analysis")
        print("USAGE:")
        print("  python integrated_simulation.py                 # Run indefinitely")
        print("  python integrated_simulation.py --duration 10   # Run for 10 seconds")
        print("  python integrated_simulation.py --plot-only     # Generate plots only")
        print("="*70)

    def create_integrated_model(self, iiwa_model_path, door_model_path):
        """Create integrated model by extending the original iiwa14.xml"""
        
        # Read the original iiwa14.xml
        with open(iiwa_model_path, 'r') as f:
            iiwa_content = f.read()
        
        # Create the integrated XML by modifying the iiwa14.xml
        integrated_xml = self.modify_iiwa_for_integration(iiwa_content)
        
        # Save the integrated XML file
        integrated_xml_path = "integrated_model.xml"
        with open(integrated_xml_path, 'w') as f:
            f.write(integrated_xml)
        
        try:
            print(f"Loading integrated model: {integrated_xml_path}")
            self.model = mujoco.MjModel.from_xml_path(integrated_xml_path)
            self.data = mujoco.MjData(self.model)
            print("✓ Integrated model loaded successfully!")
        except Exception as e:
            print(f"❌ Error loading integrated model: {e}")
            raise e

    def modify_iiwa_for_integration(self, iiwa_content):
        """Modify the iiwa14.xml to add zero gravity, gripper, and door system"""
        
        # Replace the header and add zero gravity, and update mesh paths
        integrated_content = iiwa_content.replace(
            '<mujoco model="iiwa14">',
            '<mujoco model="integrated_iiwa14">'
        ).replace(
            '<option integrator="implicitfast"/>',
            '<option integrator="implicitfast" gravity="0 0 0" timestep="0.002"/>'
        )
        
        # Update all iiwa mesh file paths to include the directory
        mesh_files = ['link_0.obj', 'link_1.obj', 'link_2_orange.obj', 'link_2_grey.obj', 
                     'link_3.obj', 'band.obj', 'kuka.obj', 'link_4_orange.obj', 
                     'link_4_grey.obj', 'link_5.obj', 'link_6_orange.obj', 
                     'link_6_grey.obj', 'link_7.obj']
        
        for mesh_file in mesh_files:
            integrated_content = integrated_content.replace(
                f'file="{mesh_file}"',
                f'file="kuka_iiwa_14/assets/{mesh_file}"'
            )
        
        # Remove the meshdir since we're using absolute paths
        integrated_content = integrated_content.replace('meshdir="assets"', '')
        
        # Add additional materials for gripper and door after existing materials
        additional_materials = '''
    <!-- Robotiq 2f85 materials -->
    <material name="gripper_metal" rgba="0.58 0.58 0.58 1"/>
    <material name="gripper_silicone" rgba="0.1882 0.1882 0.1882 1"/>
    <material name="gripper_gray" rgba="0.4627 0.4627 0.4627 1"/>
    <material name="gripper_black" rgba="0.149 0.149 0.149 1"/>
    
    <!-- Box/Door materials -->
    <material name="mat_box" rgba="0.7 0.5 0.3 1"/>
    <material name="mat_door" rgba="0.3 0.5 0.7 1"/>
    <material name="mat_walls" rgba="0.8 0.8 0.8 1"/>

    <!-- Robotiq 2f85 meshes -->
    <mesh name="base_mount" file="robotiq_2f85/assets/base_mount.stl" scale="0.001 0.001 0.001"/>
    <mesh name="base" file="robotiq_2f85/assets/base.stl" scale="0.001 0.001 0.001"/>
    <mesh name="driver" file="robotiq_2f85/assets/driver.stl" scale="0.001 0.001 0.001"/>
    <mesh name="coupler" file="robotiq_2f85/assets/coupler.stl" scale="0.001 0.001 0.001"/>
    <mesh name="follower" file="robotiq_2f85/assets/follower.stl" scale="0.001 0.001 0.001"/>
    <mesh name="pad" file="robotiq_2f85/assets/pad.stl" scale="0.001 0.001 0.001"/>
    <mesh name="silicone_pad" file="robotiq_2f85/assets/silicone_pad.stl" scale="0.001 0.001 0.001"/>
    <mesh name="spring_link" file="robotiq_2f85/assets/spring_link.stl" scale="0.001 0.001 0.001"/>

    <!-- Box/Door meshes -->
    <mesh name="Box" file="Box.stl"/>
    <mesh name="Door" file="Door.stl"/>

    <!-- Environment Meshes -->
    <mesh name="ISSDestiny" file="ISSArea1.stl" scale="0.1 0.1 0.1" />

    <!-- Environment textures -->
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
        '''
        
        # Insert additional materials before </asset>
        integrated_content = integrated_content.replace('</asset>', additional_materials + '\n  </asset>')
        
        # Add gripper defaults after iiwa defaults - find the proper location
        gripper_defaults = '''
    <default class="2f85">
      <general biastype="affine"/>
      <joint axis="1 0 0"/>
      <default class="driver">
        <joint range="0 0.8" armature="0.005" damping="0.1" solimplimit="0.95 0.99 0.001" solreflimit="0.005 1"/>
      </default>
      <default class="follower">
        <joint range="-0.872664 0.872664" armature="0.001" pos="0 -0.018 0.0065" solimplimit="0.95 0.99 0.001" solreflimit="0.005 1"/>
      </default>
      <default class="spring_link">
        <joint range="-0.29670597283 0.8" armature="0.001" stiffness="0.05" springref="2.62" damping="0.00125"/>
      </default>
      <default class="coupler">
        <joint range="-1.57 0" armature="0.001" solimplimit="0.95 0.99 0.001" solreflimit="0.005 1"/>
      </default>
      <default class="gripper_visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2"/>
      </default>
      <default class="gripper_collision">
        <geom type="mesh" group="3"/>
        <default class="pad_box1">
          <geom mass="0" type="box" pos="0 -0.0026 0.028125" size="0.011 0.004 0.009375" friction="0.7"
            solimp="0.95 0.99 0.001" solref="0.004 1" priority="1" rgba="0.55 0.55 0.55 1"/>
        </default>
        <default class="pad_box2">
          <geom mass="0" type="box" pos="0 -0.0026 0.009375" size="0.011 0.004 0.009375" friction="0.6"
            solimp="0.95 0.99 0.001" solref="0.004 1" priority="1" rgba="0.45 0.45 0.45 1"/>
        </default>
      </default>
    </default>
        '''
        
        # Find the end of the iiwa default class and insert gripper defaults after it
        # Look for the end of the main iiwa class (after site definition)
        site_pattern = '<site size="0.001" rgba="1 0 0 1" group="4"/>'
        site_pos = integrated_content.find(site_pattern)
        if site_pos != -1:
            # Find the next </default> after the site
            next_default_end = integrated_content.find('</default>', site_pos)
            if next_default_end != -1:
                next_default_end += len('</default>')
                integrated_content = (integrated_content[:next_default_end] + 
                                    '\n' + gripper_defaults + 
                                    integrated_content[next_default_end:])
        
        # Add environment and extensions to worldbody (before the last </body> </worldbody>)
        environment_and_extensions = '''
    
    <!-- Environment Container -->
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    
    <!-- Cylindrical Container made of box walls for reliable collision -->
    <!-- Floor -->
    <geom name="floor" type="box" size="4 1.3 0.1" pos="0 0.5 -0.1" material="mat_walls" contype="1" conaffinity="1"/>
    <!-- Ceiling -->  
    <geom name="ceiling" type="box" size="4 1.3 0.1" pos="0 0.5 2.0" material="mat_walls" contype="1" conaffinity="1"/>
    <!-- Walls -->
    <geom name="wall_x_pos" type="box" size="0.1 1.3 1" pos="4 0.5 1" material="mat_walls" contype="1" conaffinity="1"/>
    <geom name="wall_x_neg" type="box" size="0.1 1.3 1" pos="-2.2 0.5 1" material="mat_walls" contype="1" conaffinity="1"/>
    <geom name="wall_y_pos" type="box" size="4 0.1 1" pos="0 1.8 1" material="mat_walls" contype="1" conaffinity="1"/>
    <geom name="wall_y_neg" type="box" size="4 0.1 1" pos="0 -0.6 1" material="mat_walls" contype="1" conaffinity="1"/>

    <!-- contype and conaffinity set to 1 for walls to enable collision with everything, if just aesthetic leave 0 -->
    <geom name="ISSDestiny_geom" type="mesh" mesh="ISSDestiny" pos="-5.0 -2.8 1.1" contype="0" conaffinity="0" material="mat_walls"/> 
        '''
        
        # Find where to insert gripper (after attachment_site)
        attachment_site_pos = integrated_content.find('<site pos="0 0 0.045" name="attachment_site"/>')
        if attachment_site_pos != -1:
            # Insert gripper after the attachment site
            gripper_xml = '''
                    <!-- Gripper center site for positioning -->
                    <geom class="visual" type="sphere" size="0.005" pos="0 0 0.35" rgba="1 0 1 1"/>
                    <site name="gripper_center" pos="0 0 0.35" size="0.5" rgba="1 0 1 1"/>
                    
                    <!-- Robotiq 2f85 Gripper mounted on attachment_site -->
                    <body name="gripper_base_mount" pos="0 0 0.045" childclass="2f85">
                      <geom class="gripper_visual" mesh="base_mount" material="gripper_black"/>
                      <geom class="gripper_collision" mesh="base_mount"/>
                      
                      
                      <body name="gripper_base" pos="0 0 0.0038" quat="1 0 0 -1">
                        <geom class="gripper_visual" mesh="base" material="gripper_black"/>
                        <geom class="gripper_collision" mesh="base"/>
                        
                        <!-- Right finger assembly -->
                        <body name="right_driver" pos="0 0.0306011 0.054904">
                          <joint name="right_driver_joint" class="driver"/>
                          <geom class="gripper_visual" mesh="driver" material="gripper_gray"/>
                          <geom class="gripper_collision" mesh="driver"/>
                          
                          <body name="right_coupler" pos="0 0.0315 -0.0041">
                            <joint name="right_coupler_joint" class="coupler"/>
                            <geom class="gripper_visual" mesh="coupler" material="gripper_black"/>
                            <geom class="gripper_collision" mesh="coupler"/>
                          </body>
                        </body>
                        
                        <body name="right_spring_link" pos="0 0.0132 0.0609">
                          <joint name="right_spring_link_joint" class="spring_link"/>
                          <geom class="gripper_visual" mesh="spring_link" material="gripper_black"/>
                          <geom class="gripper_collision" mesh="spring_link"/>
                          
                          <body name="right_follower" pos="0 0.055 0.0375">
                            <joint name="right_follower_joint" class="follower"/>
                            <geom class="gripper_visual" mesh="follower" material="gripper_black"/>
                            <geom class="gripper_collision" mesh="follower"/>
                            
                            <body name="right_pad" pos="0 -0.0189 0.01352">
                              <geom class="pad_box1" name="right_pad1"/>
                              <geom class="pad_box2" name="right_pad2"/>
                              <geom class="gripper_visual" mesh="pad"/>
                              
                              <body name="right_silicone_pad">
                                <geom class="gripper_visual" mesh="silicone_pad" material="gripper_black"/>
                              </body>
                            </body>
                          </body>
                        </body>
                        
                        <!-- Left finger assembly -->
                        <body name="left_driver" pos="0 -0.0306011 0.054904" quat="0 0 0 1">
                          <joint name="left_driver_joint" class="driver"/>
                          <geom class="gripper_visual" mesh="driver" material="gripper_gray"/>
                          <geom class="gripper_collision" mesh="driver"/>
                          
                          <body name="left_coupler" pos="0 0.0315 -0.0041">
                            <joint name="left_coupler_joint" class="coupler"/>
                            <geom class="gripper_visual" mesh="coupler" material="gripper_black"/>
                            <geom class="gripper_collision" mesh="coupler"/>
                          </body>
                        </body>
                        
                        <body name="left_spring_link" pos="0 -0.0132 0.0609" quat="0 0 0 1">
                          <joint name="left_spring_link_joint" class="spring_link"/>
                          <geom class="gripper_visual" mesh="spring_link" material="gripper_black"/>
                          <geom class="gripper_collision" mesh="spring_link"/>
                          
                          <body name="left_follower" pos="0 0.055 0.0375">
                            <joint name="left_follower_joint" class="follower"/>
                            <geom class="gripper_visual" mesh="follower" material="gripper_black"/>
                            <geom class="gripper_collision" mesh="follower"/>
                            
                            <body name="left_pad" pos="0 -0.0189 0.01352">
                              <geom class="pad_box1" name="left_pad1"/>
                              <geom class="pad_box2" name="left_pad2"/>
                              <geom class="gripper_visual" mesh="pad"/>
                              
                              <body name="left_silicone_pad">
                                <geom class="gripper_visual" mesh="silicone_pad" material="gripper_black"/>
                              </body>
                            </body>
                          </body>
                        </body>

                      </body>
                    </body>
            '''
            
            # Find the end of the attachment_site line
            line_end = integrated_content.find('\n', attachment_site_pos)
            integrated_content = (integrated_content[:line_end] + 
                                gripper_xml + 
                                integrated_content[line_end:])
        
        # Add box/door assembly before </worldbody>
        box_door_assembly = '''
    
    <!-- Box/Door Assembly (floating in zero gravity, positioned closer to robot) -->
    <body name="box_door_assembly" pos="0.0 0.4 1.2" quat="0.0 1.0 1.0 0">
      <freejoint name="assembly_freejoint"/>
      
      <!-- Base Box -->
      <body name="Base_Box" pos="0 0 0">
        <!-- inertial pos="0.25 0.25 0.25" mass="2.0" diaginertia="0.05 0.05 0.05"-->
        <geom name="box_visual" type="mesh" mesh="Box" material="mat_box" contype="0" conaffinity="0" group="2"/>
        <geom name="box_collision" type="box" size="0.25 0.25 0.25" pos="0.25 0.25 0.25" contype="1" conaffinity="1" friction="0.7 0.1 0.1" material="mat_box"/>

        <!-- 3D Frame marker for target position and orientation -->
        <body name="target_frame" pos="0.075 -0.02 0.165">
          <!-- X-axis (red) -->
          <geom name="frame_x_axis" type="capsule" size="0.005 0.1" pos="0.1 0 0" rgba="1 0 0 0.8" 
                quat="0.707 0 0.707 0" contype="0" conaffinity="0"/>
          <!-- Y-axis (green) -->  
          <geom name="frame_y_axis" type="capsule" size="0.005 0.1" pos="0 0.1 0" rgba="0 1 0 0.8" 
                quat="0.707 -0.707 0 0" contype="0" conaffinity="0"/>
          <!-- Z-axis (blue) -->
          <geom name="frame_z_axis" type="capsule" size="0.005 0.1" pos="0 0 0.1" rgba="0 0 1 0.8" 
                contype="0" conaffinity="0"/>
        </body>
    
        <!-- Hinged Door -->
        <body name="Hinged_Door">
          <joint name="door_hinge" 
                 type="hinge" 
                 pos="0.075 -0.02 0.165" 
                 axis="0 0 1" 
                 range="0 1.5708"
                 damping="5.0"/>
          <geom name="door_geom" type="mesh" mesh="Door" material="mat_door" contype="1" conaffinity="1" friction="0.7 0.1 0.1"/>
          <inertial pos="0 0 0" mass="0.2" diaginertia="0.02 0.02 0.02"/>
        </body>
        <site name="door_handle_site" pos="0.075 -0.02 0.165" size="0.02" rgba="1 1 0 1"/>
      </body>
    </body>
        '''
        
        # Insert environment and box/door before </worldbody>
        worldbody_end = integrated_content.find('</worldbody>')
        if worldbody_end != -1:
            integrated_content = (integrated_content[:worldbody_end] + 
                                environment_and_extensions +
                                box_door_assembly + 
                                '\n  </worldbody>' +
                                integrated_content[worldbody_end + len('</worldbody>'):])
        
        # Add gripper tendons, actuators, and sensors after </worldbody>
        additional_components = '''

  <!-- Tendons for gripper mechanism -->
  <tendon>
    <fixed name="split">
      <joint joint="right_driver_joint" coef="0.5"/>
      <joint joint="left_driver_joint" coef="0.5"/>
    </fixed>
  </tendon>

  <!-- Additional actuators for gripper -->
  <actuator>
    <!-- Robotiq 2f85 gripper actuator -->
    <general class="2f85" name="fingers_actuator" tendon="split" forcerange="-5 5" ctrlrange="0 255"
      gainprm="0.3137255 0 0" biasprm="0 -100 -10"/>
  </actuator>

  <!-- Equality constraints for gripper mechanism -->
  <equality>
    <connect anchor="0 0 0" body1="right_follower" body2="right_coupler" solimp="0.95 0.99 0.001" solref="0.005 1"/>
    <connect anchor="0 0 0" body1="left_follower" body2="left_coupler" solimp="0.95 0.99 0.001" solref="0.005 1"/>
  </equality>

  <sensor>
    <jointpos name="door_angle" joint="door_hinge"/>
  </sensor>
        '''
        
        # Remove the keyframe for now to avoid size mismatch issues
        keyframe_start = integrated_content.find('<keyframe>')
        keyframe_end = integrated_content.find('</keyframe>') + len('</keyframe>')
        if keyframe_start != -1 and keyframe_end != -1:
            integrated_content = integrated_content[:keyframe_start] + integrated_content[keyframe_end:]
        
        # Insert before </mujoco>
        mujoco_end = integrated_content.find('</mujoco>')
        if mujoco_end != -1:
            integrated_content = (integrated_content[:mujoco_end] + 
                                additional_components + 
                                '\n</mujoco>')
        else:
            integrated_content += additional_components + '\n</mujoco>'
        
        return integrated_content

    def setup_control_parameters(self):
        """Setup joint names and control parameters using original iiwa structure"""
        
        # Robot arm joints (from original iiwa14)
        self.arm_joint_names = [
            "joint1", "joint2", "joint3", "joint4", 
            "joint5", "joint6", "joint7"
        ]
        
        # Get joint IDs for applying forces/torques directly
        self.arm_joint_ids = []
        for name in self.arm_joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            self.arm_joint_ids.append(joint_id)
        
        # Gripper actuator (Robotiq 2F85 uses tendon system)
        try:
            self.gripper_actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, "fingers_actuator")
        except:
            self.gripper_actuator_id = -1
        
        # Door joint
        self.door_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "door_hinge")
        
        # Assembly freejoint
        self.assembly_freejoint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "assembly_freejoint")
        
        print(f"✓ Found {len(self.arm_joint_ids)} arm joints")
        print(f"✓ Found gripper actuator: {'Yes' if self.gripper_actuator_id >= 0 else 'No'}")
        print(f"✓ Found door joint: {'Yes' if self.door_joint_id >= 0 else 'No'}")

    def get_end_effector_position(self):
        """Get current end-effector position"""
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripper_center")
        return self.data.site_xpos[site_id]

    def compute_jacobian(self):
        """Compute the Jacobian matrix for the robot arm using original iiwa structure"""
        # Get end effector site ID
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripper_center")
        
        if site_id >= 0:
            # Allocate Jacobian matrices
            jac_pos = np.zeros((3, self.model.nv))  # Position Jacobian
            jac_rot = np.zeros((3, self.model.nv))  # Rotation Jacobian
            
            # Compute Jacobians
            mujoco.mj_jacSite(self.model, self.data, jac_pos, jac_rot, site_id)
            
            # Combine position and rotation Jacobians for 6DOF control
            jacobian = np.vstack([jac_pos, jac_rot])
            
            # Extract only the arm joints (first 7 DOF for iiwa14)
            return jacobian[:, :7]
        else:
            # Return identity if site not found
            return np.eye(6, 7)
    
    def position_controller(self):
        """Enhanced position controller using original iiwa14 structure"""
        
        if not self.controller_enabled:
            return
            
        # 1. Get current position and orientation
        current_pos = self.get_end_effector_position()
        
        # Get current end effector orientation as rotation matrix
        orient_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripper_center")
        current_orient_mat = self.data.site_xmat[orient_id].reshape(3,3)

        # Get target position from door handle site (consistent approach)
        handle_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'door_handle_site')
        if handle_site_id >= 0:
            self.target_position = self.data.site_xpos[handle_site_id]
        else:
            self.target_position = np.array([1.3, 0.5, 0.5])
        
        # Get target orientation from box using consistent quaternion approach
        box_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box_door_assembly")
        if box_id >= 0:
            target_quat = np.array([-1, 0, -1, 0]) * self.data.xquat[box_id].copy()
            # Check for valid quaternion (non-zero norm)
            if np.linalg.norm(target_quat) > 1e-6:
                target_orient_mat = Rotation.from_quat(target_quat[[1, 2, 3, 0]]).as_matrix()
            else:
                target_orient_mat = np.eye(3)
        else:
            target_orient_mat = np.eye(3)

        # Get current joint velocities for the arm (first 7 joints)
        current_joint_vel = self.data.qvel[:7]  # First 7 DOF are arm joints
        
        # Get current angular velocity of end effector in world frame
        J_full = self.compute_jacobian()  # This gives us 6x7 jacobian
        J_rot = J_full[3:6, :7]  # Rotational part of Jacobian
        current_angular_vel = J_rot @ current_joint_vel  # Angular velocity in world frame
        
        # 2. Calculate position error
        position_error = self.target_position - current_pos
        
        # 3. Calculate orientation error using rotation matrices (fixed direction)
        R_error = current_orient_mat.T @ target_orient_mat
        
        # Convert rotation matrix error to axis-angle representation
        orientation_error = np.array([
            R_error[2, 1] - R_error[1, 2],
            R_error[0, 2] - R_error[2, 0],
            R_error[1, 0] - R_error[0, 1]
        ]) * 0.5
        
        # 4. Calculate angular velocity error
        target_angular_vel = self.data.qvel[3:6] if self.model.nv > 6 else np.zeros(3)
        angular_vel_error = target_angular_vel - current_angular_vel
        
        # 5. Get Jacobian (position part only, 3x7)
        J = self.compute_jacobian()
        J_pos = J[:3, :7]  # Take only position Jacobian for arm joints
        
        # 6. Controller gains (velocity control - enhanced Z control and proper orientation)
        K_pos = np.diag([8.2, 10.2, 7.0]) * 1.0         # Position velocity gains
        K_angular_vel = np.diag([1.0, 1.0, 1.0]) * 0.05    # Angular velocity damping gain matrix

        # 7. Compute desired Cartesian velocities (PD control in Cartesian space)
        desired_position_velocity = K_pos @ position_error
        desired_angular_velocity = K_angular_vel @ angular_vel_error
        
        # Combine position and angular velocities for 6DOF control
        desired_cartesian_velocity = np.concatenate([desired_position_velocity, desired_angular_velocity])
        
        # 8. Map to joint space using redundant manipulator control with null space projection
        try:
            lambda_damping = 0.25  # Damping factor for numerical stability
            J_damped_pinv = J.T @ np.linalg.inv(J @ J.T + lambda_damping * np.eye(6))
            joint_velocities = J_damped_pinv @ desired_cartesian_velocity
        except np.linalg.LinAlgError:
            joint_velocities = np.zeros(7)
        
        # 9. Apply velocity commands with limits
        max_velocity = 4.0  # Maximum velocity limit (rad/s)
        joint_velocities_clipped = np.clip(joint_velocities, -max_velocity, max_velocity)
        
        # Set velocity commands directly to ctrl (velocity control)
        self.data.ctrl[:len(joint_velocities_clipped)] = joint_velocities_clipped
        if self.model.nu > len(joint_velocities_clipped):
            self.data.ctrl[len(joint_velocities_clipped):] = 0.0
        
        # 10. Store target orientation for frame visualization
        self.current_target_orientation = target_orient_mat.copy()
        
        # 11. Collect data for plotting
        self.collect_controller_data(current_pos, current_orient_mat, target_orient_mat, 
                                   position_error, orientation_error, current_joint_vel, current_angular_vel)
        
        # 12. Update previous error for next iteration
        self.previous_position_error = position_error.copy()

    def collect_controller_data(self, current_pos, current_orient_mat, target_orient_mat, 
                              position_error, orientation_error, current_joint_vel, current_angular_vel):
        """Collect comprehensive controller data for analysis"""
        
        if self.simulation_start_time is None:
            self.simulation_start_time = time.time()
        
        current_time = time.time() - self.simulation_start_time
        
        # Convert rotation matrices to quaternions for storage
        current_quat = Rotation.from_matrix(current_orient_mat).as_quat()  # [x, y, z, w]
        target_quat = Rotation.from_matrix(target_orient_mat).as_quat()    # [x, y, z, w]
        
        # Ensure quaternion consistency (avoid sign flips)
        if len(self.current_orientation_data) > 0:
            prev_quat = self.current_orientation_data[-1]
            if np.dot(current_quat, prev_quat) < 0:
                current_quat = -current_quat
        
        if len(self.target_orientation_data) > 0:
            prev_target_quat = self.target_orientation_data[-1]
            if np.dot(target_quat, prev_target_quat) < 0:
                target_quat = -target_quat
        
        self.time_data.append(current_time)
        self.position_error_data.append(position_error.copy())
        self.target_position_data.append(self.target_position.copy())
        self.orientation_error_data.append(orientation_error.copy())
        self.current_position_data.append(current_pos.copy())
        self.current_orientation_data.append(current_quat.copy())
        self.target_orientation_data.append(target_quat.copy())
        self.joint_values_data.append(self.data.qpos[:7].copy())
        self.joint_velocities_data.append(current_joint_vel.copy())
        
        # Calculate end effector linear velocity
        if len(self.current_position_data) >= 2:
            dt = self.time_data[-1] - self.time_data[-2]
            if dt > 0:
                linear_vel = (self.current_position_data[-1] - self.current_position_data[-2]) / dt
            else:
                linear_vel = np.zeros(3)
        else:
            linear_vel = np.zeros(3)
        self.linear_velocity_data.append(linear_vel.copy())
        self.angular_velocity_data.append(current_angular_vel.copy())

    def plot_controller_data(self):
        """Generate comprehensive plots of controller performance"""
        
        if len(self.time_data) < 10:
            print("⚠️ Not enough data collected for plotting (need at least 10 data points)")
            return
        
        print("📊 Generating plots from collected data...")
        
        # Convert deques to numpy arrays for plotting
        time_array = np.array(self.time_data)
        pos_error_array = np.array(self.position_error_data)
        target_pos_array = np.array(self.target_position_data)
        current_pos_array = np.array(self.current_position_data)
        orient_error_array = np.array(self.orientation_error_data)
        current_orient_array = np.array(self.current_orientation_data)
        target_orient_array = np.array(self.target_orientation_data)
        joint_values_array = np.array(self.joint_values_data)
        joint_vel_array = np.array(self.joint_velocities_data)
        linear_vel_array = np.array(self.linear_velocity_data)
        angular_vel_array = np.array(self.angular_velocity_data)
        
        # Calculate position error magnitude
        pos_error_mag = np.linalg.norm(pos_error_array, axis=1)
        
        # Create comprehensive plot with 3x3 subplots
        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        fig.suptitle('Integrated KUKA iiwa14 Controller Performance Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Position Error Magnitude over Time
        axes[0, 0].plot(time_array, pos_error_mag, 'r-', linewidth=2, label='Position Error Magnitude')
        axes[0, 0].axhline(y=0.1, color='g', linestyle='--', alpha=0.7, label='Target (0.1m)')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Position Error (m)')
        axes[0, 0].set_title('Position Error Magnitude')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot 2: Position Error Components
        axes[0, 1].plot(time_array, pos_error_array[:, 0], 'r-', label='X Error')
        axes[0, 1].plot(time_array, pos_error_array[:, 1], 'g-', label='Y Error') 
        axes[0, 1].plot(time_array, pos_error_array[:, 2], 'b-', label='Z Error')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Position Error (m)')
        axes[0, 1].set_title('Position Error Components')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot 3: Current vs Target Position (3D components)
        axes[0, 2].plot(time_array, current_pos_array[:, 0], 'r-', label='Current X')
        axes[0, 2].plot(time_array, target_pos_array[:, 0], 'r--', alpha=0.7, label='Target X')
        axes[0, 2].plot(time_array, current_pos_array[:, 1], 'g-', label='Current Y')
        axes[0, 2].plot(time_array, target_pos_array[:, 1], 'g--', alpha=0.7, label='Target Y')
        axes[0, 2].plot(time_array, current_pos_array[:, 2], 'b-', label='Current Z')
        axes[0, 2].plot(time_array, target_pos_array[:, 2], 'b--', alpha=0.7, label='Target Z')
        axes[0, 2].set_xlabel('Time (s)')
        axes[0, 2].set_ylabel('Position (m)')
        axes[0, 2].set_title('Current vs Target Position')
        axes[0, 2].legend()
        axes[0, 2].grid(True)
        
        # Plot 4: Orientation Error Components  
        axes[1, 0].plot(time_array, orient_error_array[:, 0], 'r-', label='X Rotation Error')
        axes[1, 0].plot(time_array, orient_error_array[:, 1], 'g-', label='Y Rotation Error')
        axes[1, 0].plot(time_array, orient_error_array[:, 2], 'b-', label='Z Rotation Error')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Orientation Error (rad)')
        axes[1, 0].set_title('Orientation Error Components')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot 5: Current vs Target Orientation (Quaternions) - All 4 components
        axes[1, 1].plot(time_array, current_orient_array[:, 0], 'r-', label='Current qx')
        axes[1, 1].plot(time_array, target_orient_array[:, 0], 'r--', alpha=0.7, label='Target qx')
        axes[1, 1].plot(time_array, current_orient_array[:, 1], 'g-', label='Current qy')
        axes[1, 1].plot(time_array, target_orient_array[:, 1], 'g--', alpha=0.7, label='Target qy')
        axes[1, 1].plot(time_array, current_orient_array[:, 2], 'b-', label='Current qz')
        axes[1, 1].plot(time_array, target_orient_array[:, 2], 'b--', alpha=0.7, label='Target qz')
        axes[1, 1].plot(time_array, current_orient_array[:, 3], 'm-', label='Current qw')
        axes[1, 1].plot(time_array, target_orient_array[:, 3], 'm--', alpha=0.7, label='Target qw')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Quaternion Components')
        axes[1, 1].set_title('Current vs Target Orientation (Quaternions)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        # Plot 6: Joint Values
        axes[1, 2].clear()
        for i in range(min(7, joint_values_array.shape[1])):
            axes[1, 2].plot(time_array, joint_values_array[:, i], label=f'Joint {i+1}')
        axes[1, 2].set_xlabel('Time (s)')
        axes[1, 2].set_ylabel('Joint Angles (rad)')
        axes[1, 2].set_title('Joint Values')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        
        # Plot 7: Joint Velocities
        axes[2, 0].clear()
        for i in range(min(7, joint_vel_array.shape[1])):
            axes[2, 0].plot(time_array, joint_vel_array[:, i], label=f'Joint {i+1}')
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 0].set_ylabel('Joint Velocities (rad/s)')
        axes[2, 0].set_title('Joint Velocities')
        axes[2, 0].legend()
        axes[2, 0].grid(True)
        
        # Plot 8: End Effector Linear Velocity
        axes[2, 1].plot(time_array, linear_vel_array[:, 0], 'r-', label='Vx')
        axes[2, 1].plot(time_array, linear_vel_array[:, 1], 'g-', label='Vy')
        axes[2, 1].plot(time_array, linear_vel_array[:, 2], 'b-', label='Vz')
        axes[2, 1].set_xlabel('Time (s)')
        axes[2, 1].set_ylabel('Linear Velocity (m/s)')
        axes[2, 1].set_title('End Effector Linear Velocity')
        axes[2, 1].legend()
        axes[2, 1].grid(True)
        
        # Plot 9: End Effector Angular Velocity
        axes[2, 2].plot(time_array, angular_vel_array[:, 0], 'r-', label='ωx')
        axes[2, 2].plot(time_array, angular_vel_array[:, 1], 'g-', label='ωy')
        axes[2, 2].plot(time_array, angular_vel_array[:, 2], 'b-', label='ωz')
        axes[2, 2].set_xlabel('Time (s)')
        axes[2, 2].set_ylabel('Angular Velocity (rad/s)')
        axes[2, 2].set_title('End Effector Angular Velocity')
        axes[2, 2].legend()
        axes[2, 2].grid(True)
        
        plt.tight_layout()
        plt.savefig('integrated_controller_performance_analysis.png', dpi=300, bbox_inches='tight')
        print("📊 Integrated controller analysis saved to 'integrated_controller_performance_analysis.png'!")
        plt.show()

    def run_simulation(self, duration=None):
        """Run the integrated simulation with viewer"""
        
        if duration is not None:
            print(f"🚀 Starting integrated simulation for {duration} seconds...")
        else:
            print("🚀 Starting integrated simulation (press ESC to exit)...")
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            start_time = time.time()
            step_count = 0
            #viewer.cam.lookat[:] = [0.0, 3.725, 1.5]
            
            viewer.cam.lookat[:]=[0.0,0.5,1.0]
            viewer.cam.distance=2.249
            viewer.cam.azimuth=0.75
            viewer.cam.elevation=-20.0
            # Initialize to default position
            mujoco.mj_resetData(self.model, self.data)
            
            while viewer.is_running():
                step_start = time.time()
                
                # Run controller
                self.position_controller()
                
                # Step simulation
                mujoco.mj_step(self.model, self.data)
                
                # Update viewer
                viewer.sync()
                
                # Print status every 60 steps (approximately every second at 60Hz)
                if step_count % 60 == 0:
                    current_pos = self.get_end_effector_position()
                    if hasattr(self, 'target_position'):
                        error = np.linalg.norm(self.target_position - current_pos)
                        print(f"End-Effector: [{current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f}] | "
                              f"Target: [{self.target_position[0]:.2f}, {self.target_position[1]:.2f}, {self.target_position[2]:.2f}] | "
                              f"Error: {error:.3f}m | Controller: {'ON' if self.controller_enabled else 'OFF'} | "
                              f"Gripper: {int(self.gripper_target)}/255")
                
                # Check duration limit
                if duration is not None and (time.time() - start_time) >= duration:
                    break
                
                step_count += 1
                
                # Maintain real-time execution
                time_until_next_step = self.model.opt.timestep - (time.time() - step_start)
                if time_until_next_step > 0:
                    time.sleep(time_until_next_step)
        
        print("👋 Simulation ended")
        self.plot_controller_data()


def main():
    """Main function"""
    
    parser = argparse.ArgumentParser(description='Run the integrated zero-gravity simulation')
    parser.add_argument('--duration', '-d', type=float, default=None, 
                        help='Duration to run simulation in seconds (default: run indefinitely)')
    parser.add_argument('--plot-only', action='store_true',
                        help='Skip simulation, just generate plots from existing data')
    
    args = parser.parse_args()
    
    simulation = None
    try:
        if args.plot_only:
            print("📊 Plot-only mode: Loading existing simulation data...")
            # This would load existing data if available
            print("⚠️ No existing data found. Run simulation first to generate data.")
            return
        
        simulation = IntegratedZeroGravitySimulation()
        simulation.run_simulation(duration=args.duration)
        
    except KeyboardInterrupt:
        print("\n👋 Simulation interrupted by user")
    except Exception as e:
        print(f"❌ Simulation error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if simulation and hasattr(simulation, 'model'):
            print("🧹 Cleaning up simulation resources...")


if __name__ == "__main__":
    main()
