#!/usr/bin/env python3
"""
Combined Zero-Gravity Simulation: Kuka Robot with Gripper + Box/Door Model

This simulation loads both the Kuka robot with gripper and the box/door model
in a zero-gravity environment. The models are not connected and float freely.

Mouse controls are available in the viewer for camera movement.
"""

import mujoco
import mujoco.viewer
import numpy as np
import time
import sys
import os
import matplotlib.pyplot as plt
from collections import deque

class CombinedZeroGravitySimulation:
    def __init__(self, 
                 robot_model_path="kuka_simple_gripper.xml", 
                 door_model_path="door_hinge_model.xml"):
        """Initialize the combined simulation with robot and door models"""
        
        # Check if model files exist
        if not os.path.exists(robot_model_path):
            print(f"❌ Robot model file not found: {robot_model_path}")
            sys.exit(1)
            
        if not os.path.exists(door_model_path):
            print(f"❌ Door model file not found: {door_model_path}")
            sys.exit(1)

        #######---------------------------------------------------------------------------
        # Create combined MJCF model

        # This is where i assemble the xml of the other two models
        ############################################################################
        
        self.create_combined_model(robot_model_path, door_model_path)
        
        # Initialize control parameters
        self.setup_control_parameters()
        
        # Movement step sizes
        self.position_step = 0.02  # meters
        self.position_step_fast = 0.05  # faster movement
        self.impulse_strength = 2.0  # impulse strength for box movement
        
        # =================================================================
        # CONTROLLER EXERCISE VARIABLES - IMPLEMENT YOUR CONTROLLER BELOW
        # =================================================================
        
        
        
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
        print("🚀 ZERO-GRAVITY FREE PHYSICS SIMULATION")
        print("🤖 Kuka iiwa14 + Robotiq 2F85 Gripper + Box/Door Model")
        print("="*70)
        print("ADVANCED CONTROLLER FRAMEWORK:")
        print("  • 6DOF Cartesian space control with rotation matrix orientation")
        print("  • Redundant manipulator control with null space projection")
        print("  • Damped least squares to avoid singularities")
        print("  • Angular velocity damping for orientation stability")
        print("  • Comprehensive real-time data collection and analysis")
        print("  • Advanced plotting: errors, velocities, joint values, targets")
        print("USAGE:")
        print("  python combined_simulation.py                 # Run indefinitely")
        print("  python combined_simulation.py --duration 10   # Run for 10 seconds")
        print("  python combined_simulation.py --plot-only     # Generate plots only")
        print("="*70)

    def create_combined_model(self, robot_model_path, door_model_path):
        """Create a combined MJCF model with both robot and door in zero gravity"""
        
        # Create combined XML content
        combined_xml = f'''
<mujoco model="Combined Zero Gravity Simulation">
  <compiler angle="radian" autolimits="true"/>
  
  <option cone="elliptic" impratio="10" gravity="0 0 0" timestep="0.002"/>

  <statistic center="0 0 0" extent="12.0"/>

  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-120" elevation="-20"/>
  </visual>

  <asset>
    <!-- Kuka iiwa14 meshes -->
    <mesh name="link_0" file="kuka_iiwa_14/assets/link_0.obj"/>
    <mesh name="link_1" file="kuka_iiwa_14/assets/link_1.obj"/>
    <mesh name="link_2_grey" file="kuka_iiwa_14/assets/link_2_grey.obj"/>
    <mesh name="link_2_orange" file="kuka_iiwa_14/assets/link_2_orange.obj"/>
    <mesh name="link_3" file="kuka_iiwa_14/assets/link_3.obj"/>
    <mesh name="band" file="kuka_iiwa_14/assets/band.obj"/>
    <mesh name="kuka" file="kuka_iiwa_14/assets/kuka.obj"/>
    <mesh name="link_4_grey" file="kuka_iiwa_14/assets/link_4_grey.obj"/>
    <mesh name="link_4_orange" file="kuka_iiwa_14/assets/link_4_orange.obj"/>
    <mesh name="link_5" file="kuka_iiwa_14/assets/link_5.obj"/>
    <mesh name="link_6_grey" file="kuka_iiwa_14/assets/link_6_grey.obj"/>
    <mesh name="link_6_orange" file="kuka_iiwa_14/assets/link_6_orange.obj"/>
    <mesh name="link_7" file="kuka_iiwa_14/assets/link_7.obj"/>

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

    <!-- Materials -->
    <material name="kuka_grey" rgba="0.7 0.7 0.7 1"/>
    <material name="kuka_orange" rgba="1.0 0.423 0.039 1"/>
    <material name="gripper_metal" rgba="0.58 0.58 0.58 1"/>
    <material name="gripper_silicone" rgba="0.1882 0.1882 0.1882 1"/>
    <material name="gripper_gray" rgba="0.4627 0.4627 0.4627 1"/>
    <material name="gripper_black" rgba="0.149 0.149 0.149 1"/>
    <material name="mat_box" rgba="0.7 0.5 0.3 1"/>
    <material name="mat_door" rgba="0.3 0.5 0.7 1"/>
    <material name="mat_walls" rgba="0.8 0.8 0.8 1"/>

    <!-- Environment textures -->
    <texture type="skybox" builtin="gradient" rgb1="0.3 0.5 0.7" rgb2="0 0 0" width="512" height="3072"/>
  </asset>

  <default>
    <!-- CHECK THIS DAMPING ------------------------------------------------------------------------------------------------------- -->
    <joint axis="0 0 1" range="-3.14159 3.14159" damping="0.01"/>
    <!-- CHECK THIS DAMPING ------------------------------------------------------------------------------------------------------- -->
    <geom type="mesh" friction="1 0.1 0.001"/>
    
    <default class="iiwa">
      <material specular="0.5" shininess="0.25"/>
      <joint axis="0 0 1"/>
      <general gaintype="fixed" biastype="affine" gainprm="2000" biasprm="0 -2000 -200"/>
      <default class="joint1">
        <joint range="-2.96706 2.96706"/>
        <general ctrlrange="-2.96706 2.96706"/>
        <default class="joint2">
          <joint range="-2.0944 2.0944"/>
          <general ctrlrange="-2.0944 2.0944"/>
        </default>
      </default>
      <default class="joint3">
        <joint range="-3.05433 3.05433"/>
        <general ctrlrange="-3.05433 3.05433"/>
      </default>
      <default class="iiwa_visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2" material="kuka_grey"/>
      </default>
      <default class="iiwa_collision">
        <geom type="sphere" group="3"/>
      </default>
      <site size="0.001" rgba="1 0 0 1" group="4"/>
    </default>
    
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
      <default class="visual">
        <geom type="mesh" contype="0" conaffinity="0" group="2"/>
      </default>
      <default class="collision">
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
  </default>

  <worldbody>
    <light pos="0 0 1.5" dir="0 0 -1" directional="true"/>
    
    <!-- Cylindrical Container made of box walls for reliable collision -->
    <!-- Floor -->
    <geom name="floor" type="box" size="4 4 0.1" pos="0 0 -0.1" material="mat_walls" contype="1" conaffinity="1"/>
    <!-- Ceiling -->  
    <geom name="ceiling" type="box" size="4 4 0.1" pos="0 0 2.1" material="mat_walls" contype="1" conaffinity="1"/>
    <!-- Walls -->
    <geom name="wall_x_pos" type="box" size="0.1 4 1" pos="4 0 1" material="mat_walls" contype="1" conaffinity="1"/>
    <geom name="wall_x_neg" type="box" size="0.1 4 1" pos="-4 0 1" material="mat_walls" contype="1" conaffinity="1"/>
    <geom name="wall_y_pos" type="box" size="4 0.1 1" pos="0 4 1" material="mat_walls" contype="1" conaffinity="1"/>
    <geom name="wall_y_neg" type="box" size="4 0.1 1" pos="0 -4 1" material="mat_walls" contype="1" conaffinity="1"/>
    
    <!-- KUKA Robot with Gripper (fixed base for stability) -->
    <body name="robot_base" pos="0 0 0" childclass="iiwa">
      <inertial mass="5" pos="-0.1 0 0.07" diaginertia="0.05 0.06 0.03"/>
        <geom class="iiwa_visual" mesh="link_0"/>
        <geom class="iiwa_collision" size="0.12" pos="0 0 0.03"/>
        <geom class="iiwa_collision" size="0.08" pos="-0.08 0 0.103"/>
        <geom class="iiwa_collision" size="0.08" pos="-0.08 0 0.04"/>
        <geom class="iiwa_collision" size="0.1" pos="0 0 0.14"/>      <!-- Robot arm chain (same as original) -->
      <body name="link1" pos="0 0 0.1575">
        <inertial mass="5.76" pos="0 -0.03 0.12" diaginertia="0.0333 0.033 0.0123"/>
        <joint name="joint1" class="joint1"/>
        <geom class="iiwa_visual" mesh="link_1"/>
        <geom class="iiwa_collision" size="0.08" pos="0 0 -0.0005"/>
        <geom class="iiwa_collision" size="0.075" pos="0.01 -0.025 0.0425"/>
        <geom class="iiwa_collision" size="0.075" pos="-0.01 -0.025 0.0425"/>
        <geom class="iiwa_collision" size="0.07" pos="0.01 -0.045 0.1025"/>
        <geom class="iiwa_collision" size="0.07" pos="-0.01 -0.045 0.1025"/>
        
        <body name="link2" pos="0 0 0.2025" quat="0 0 1 1">
          <inertial mass="6.35" pos="0.0003 0.059 0.042" diaginertia="0.0305 0.0304 0.011" quat="0 0 1 1"/>
          <joint name="joint2" class="joint2"/>
          <geom class="iiwa_visual" material="kuka_orange" mesh="link_2_orange"/>
          <geom class="iiwa_visual" mesh="link_2_grey"/>
          <geom class="iiwa_collision" size="0.095" pos="0 0 -0.01"/>
          <geom class="iiwa_collision" size="0.09" pos="0 0 0.045"/>
          <geom class="iiwa_collision" size="0.07" pos="-0.01 0.04 0.054"/>
          <geom class="iiwa_collision" size="0.065" pos="-0.01 0.09 0.04"/>
          <geom class="iiwa_collision" size="0.065" pos="-0.01 0.13 0.02"/>
          <geom class="iiwa_collision" size="0.07" pos="0.01 0.04 0.054"/>
          <geom class="iiwa_collision" size="0.065" pos="0.01 0.09 0.04"/>
          <geom class="iiwa_collision" size="0.065" pos="0.01 0.13 0.02"/>
          <geom class="iiwa_collision" size="0.075" pos="0 0.18 0"/>
          
          <body name="link3" pos="0 0.2045 0" quat="0 0 1 1">
            <inertial mass="3.5" pos="0 0.03 0.13" diaginertia="0.025 0.0238 0.0076"/>
            <joint name="joint3" class="joint1"/>
            <geom class="iiwa_visual" mesh="link_3"/>
            <geom class="iiwa_visual" material="kuka_grey" mesh="band"/>
            <geom class="iiwa_visual" material="kuka_grey" mesh="kuka"/>
            <geom class="iiwa_collision" size="0.075" pos="0 0 0.0355"/>
            <geom class="iiwa_collision" size="0.06" pos="0.01 0.023 0.0855"/>
            <geom class="iiwa_collision" size="0.055" pos="0.01 0.048 0.1255"/>
            <geom class="iiwa_collision" size="0.06" pos="0.01 0.056 0.1755"/>
            <geom class="iiwa_collision" size="0.06" pos="-0.01 0.023 0.0855"/>
            <geom class="iiwa_collision" size="0.055" pos="-0.01 0.048 0.1255"/>
            <geom class="iiwa_collision" size="0.06" pos="-0.01 0.056 0.1755"/>
            <geom class="iiwa_collision" size="0.075" pos="0 0.045 0.2155"/>
            <geom class="iiwa_collision" size="0.075" pos="0 0 0.2155"/>
            
            <body name="link4" pos="0 0 0.2155" quat="1 1 0 0">
              <inertial mass="3.5" pos="0 0.067 0.034" diaginertia="0.017 0.0164 0.006" quat="1 1 0 0"/>
              <joint name="joint4" class="joint2"/>
              <geom class="iiwa_visual" material="kuka_orange" mesh="link_4_orange"/>
              <geom class="iiwa_visual" mesh="link_4_grey"/>
              <geom class="iiwa_collision" size="0.078" pos="0 0.01 0.046"/>
              <geom class="iiwa_collision" size="0.06" pos="0.01 0.06 0.052"/>
              <geom class="iiwa_collision" size="0.065" pos="0.01 0.12 0.034"/>
              <geom class="iiwa_collision" size="0.06" pos="-0.01 0.06 0.052"/>
              <geom class="iiwa_collision" size="0.065" pos="-0.01 0.12 0.034"/>
              <geom class="iiwa_collision" size="0.075" pos="0 0.184 0"/>
              
              <body name="link5" pos="0 0.1845 0" quat="0 0 1 1">
                <inertial mass="3.5" pos="0.0001 0.021 0.076" diaginertia="0.01 0.0087 0.00449"/>
                <joint name="joint5" class="joint1"/>
                <geom class="iiwa_visual" mesh="link_5"/>
                <geom class="iiwa_visual" material="kuka_grey" mesh="band"/>
                <geom class="iiwa_visual" material="kuka_grey" mesh="kuka"/>
                <geom class="iiwa_collision" size="0.075" pos="0 0 0.0335"/>
                <geom class="iiwa_collision" size="0.05" pos="-0.012 0.031 0.0755"/>
                <geom class="iiwa_collision" size="0.05" pos="0.012 0.031 0.0755"/>
                <geom class="iiwa_collision" size="0.04" pos="-0.012 0.06 0.1155"/>
                <geom class="iiwa_collision" size="0.04" pos="0.012 0.06 0.1155"/>
                <geom class="iiwa_collision" size="0.04" pos="-0.01 0.065 0.1655"/>
                <geom class="iiwa_collision" size="0.04" pos="0.01 0.065 0.1655"/>
                <geom class="iiwa_collision" size="0.035" pos="-0.012 0.065 0.1855"/>
                <geom class="iiwa_collision" size="0.035" pos="0.012 0.065 0.1855"/>
                
                <body name="link6" pos="0 0 0.2155" quat="1 1 0 0">
                  <inertial mass="1.8" pos="0 0.0006 0.0004" diaginertia="0.0049 0.0047 0.0036" quat="1 1 0 0"/>
                  <joint name="joint6" class="joint2"/>
                  <geom class="iiwa_visual" material="kuka_orange" mesh="link_6_orange"/>
                  <geom class="iiwa_visual" mesh="link_6_grey"/>
                  <geom class="iiwa_collision" size="0.055" pos="0 0 -0.059"/>
                  <geom class="iiwa_collision" size="0.065" pos="0 -0.03 0.011"/>
                  <geom class="iiwa_collision" size="0.08"/>
                  
                  <body name="link7" pos="0 0.081 0" quat="0 0 1 1">
                    <inertial mass="1.2" pos="0 0 0.02" diaginertia="0.001 0.001 0.001"/>
                    <joint name="joint7" class="joint3"/>
                    <geom class="iiwa_visual" mesh="link_7"/>
                    <geom class="iiwa_collision" size="0.06" pos="0 0 0.001"/>
                    <site pos="0 0 0.045" name="attachment_site"/>

                    <!-- Robotiq 2f85 Gripper -->
                    <body name="gripper_base_mount" pos="0 0 0.045" childclass="2f85">
                      <geom class="visual" mesh="base_mount" material="gripper_black"/>
                      <geom class="collision" mesh="base_mount"/>
                      
                      <body name="gripper_base" pos="0 0 0.0038" quat="1 0 0 -1">
                        <geom class="visual" mesh="base" material="gripper_black"/>
                        <geom class="collision" mesh="base"/>
                        
                        <!-- Right finger assembly -->
                        <body name="right_driver" pos="0 0.0306011 0.054904">
                          <joint name="right_driver_joint" class="driver"/>
                          <geom class="visual" mesh="driver" material="gripper_gray"/>
                          <geom class="collision" mesh="driver"/>
                          
                          <body name="right_coupler" pos="0 0.0315 -0.0041">
                            <joint name="right_coupler_joint" class="coupler"/>
                            <geom class="visual" mesh="coupler" material="gripper_black"/>
                            <geom class="collision" mesh="coupler"/>
                          </body>
                        </body>
                        
                        <body name="right_spring_link" pos="0 0.0132 0.0609">
                          <joint name="right_spring_link_joint" class="spring_link"/>
                          <geom class="visual" mesh="spring_link" material="gripper_black"/>
                          <geom class="collision" mesh="spring_link"/>
                          
                          <body name="right_follower" pos="0 0.055 0.0375">
                            <joint name="right_follower_joint" class="follower"/>
                            <geom class="visual" mesh="follower" material="gripper_black"/>
                            <geom class="collision" mesh="follower"/>
                            
                            <body name="right_pad" pos="0 -0.0189 0.01352">
                              <geom class="pad_box1" name="right_pad1"/>
                              <geom class="pad_box2" name="right_pad2"/>
                              <geom class="visual" mesh="pad"/>
                              
                              <body name="right_silicone_pad">
                                <geom class="visual" mesh="silicone_pad" material="gripper_black"/>
                              </body>
                            </body>
                          </body>
                        </body>
                        
                        <!-- Left finger assembly -->
                        <body name="left_driver" pos="0 -0.0306011 0.054904" quat="0 0 0 1">
                          <joint name="left_driver_joint" class="driver"/>
                          <geom class="visual" mesh="driver" material="gripper_gray"/>
                          <geom class="collision" mesh="driver"/>
                          
                          <body name="left_coupler" pos="0 0.0315 -0.0041">
                            <joint name="left_coupler_joint" class="coupler"/>
                            <geom class="visual" mesh="coupler" material="gripper_black"/>
                            <geom class="collision" mesh="coupler"/>
                          </body>
                        </body>
                        
                        <body name="left_spring_link" pos="0 -0.0132 0.0609" quat="0 0 0 1">
                          <joint name="left_spring_link_joint" class="spring_link"/>
                          <geom class="visual" mesh="spring_link" material="gripper_black"/>
                          <geom class="collision" mesh="spring_link"/>
                          
                          <body name="left_follower" pos="0 0.055 0.0375">
                            <joint name="left_follower_joint" class="follower"/>
                            <geom class="visual" mesh="follower" material="gripper_black"/>
                            <geom class="collision" mesh="follower"/>
                            
                            <body name="left_pad" pos="0 -0.0189 0.01352">
                              <geom class="pad_box1" name="left_pad1"/>
                              <geom class="pad_box2" name="left_pad2"/>
                              <geom class="visual" mesh="pad"/>
                              
                              <body name="left_silicone_pad">
                                <geom class="visual" mesh="silicone_pad" material="gripper_black"/>
                              </body>
                            </body>
                          </body>
                        </body>

                        <!-- Gripper center site for positioning -->
                        <site name="gripper_center" pos="0 0 0.145" size="0.005" rgba="1 0 0 1"/>
                      </body>
                    </body>
                  </body>
                </body>
              </body>
            </body>
          </body>
        </body>
      </body>
    </body>

    <!-- Box/Door Assembly (floating in zero gravity, positioned closer to robot) -->
    <body name="box_door_assembly" pos="1.3 0.5 0.5" quat="0.0 1.0 1.0 0">
      <freejoint name="assembly_freejoint"/>
      
      <!-- Base Box -->
      <body name="Base_Box" pos="0 0 0">
        <inertial pos="0.25 0.25 0.25" mass="2.0" diaginertia="0.05 0.05 0.05"/>
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
  </worldbody>

  <contact>
    <exclude body1="robot_base" body2="link1"/>
    <exclude body1="robot_base" body2="link2"/>
    <exclude body1="robot_base" body2="link3"/>
    <exclude body1="link1" body2="link3"/>
    <exclude body1="link3" body2="link5"/>
    <exclude body1="link4" body2="link7"/>
    <exclude body1="link5" body2="link7"/>
  </contact>

  <!-- Tendons for gripper mechanism -->
  <tendon>
    <fixed name="split">
      <joint joint="right_driver_joint" coef="0.5"/>
      <joint joint="left_driver_joint" coef="0.5"/>
    </fixed>
  </tendon>

  <!-- Actuators for robot arm and gripper -->
  <actuator>
    <!-- Robot arm actuators -->
    <general name="actuator1" joint="joint1" ctrlrange="-2.96706 2.96706" gainprm="300" biasprm="0 -300 -50"/>
    <general name="actuator2" joint="joint2" ctrlrange="-2.09440 2.09440" gainprm="300" biasprm="0 -300 -50"/>
    <general name="actuator3" joint="joint3" ctrlrange="-2.96706 2.96706" gainprm="300" biasprm="0 -300 -50"/>
    <general name="actuator4" joint="joint4" ctrlrange="-2.09440 2.09440" gainprm="300" biasprm="0 -300 -50"/>
    <general name="actuator5" joint="joint5" ctrlrange="-2.96706 2.96706" gainprm="300" biasprm="0 -300 -50"/>
    <general name="actuator6" joint="joint6" ctrlrange="-2.09440 2.09440" gainprm="300" biasprm="0 -300 -50"/>
    <general name="actuator7" joint="joint7" ctrlrange="-3.05433 3.05433" gainprm="300" biasprm="0 -300 -50"/>
    
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

  <!-- Keyframes for different poses -->
  <keyframe>
    <key name="home" qpos="0 0.785398 0 -1.5708 0 0 0 0 0 0 0 0 0 0 0 2.5 1.5 0.5 1 0 0 0 0" />
    <!-- key name="home" qpos="0 0.785398 0 -1.5708 0 0 0" ctrl="0 0.785398 0 -1.5708 0 0 0"-->
  </keyframe>
</mujoco>
        '''
        
        # Save the combined XML file
        combined_xml_path = "combined_model.xml"
        with open(combined_xml_path, 'w') as f:
            f.write(combined_xml)
        
        try:
            print(f"Loading combined model: {combined_xml_path}")
            self.model = mujoco.MjModel.from_xml_path(combined_xml_path)
            self.data = mujoco.MjData(self.model)
            print("✓ Combined model loaded successfully!")
            
        except Exception as e:
            print(f"❌ Error loading combined model: {e}")
            print("Check that all mesh files and mujoco_menagerie path are correct")
            sys.exit(1)

    def setup_control_parameters(self):
        """Setup joint names and control parameters"""
        
        # Robot arm joints
        self.arm_joint_names = [
            "joint1", "joint2", "joint3", "joint4", 
            "joint5", "joint6", "joint7"
        ]
        
        # Get joint IDs for applying forces/torques directly
        self.arm_joint_ids = []
        for name in self.arm_joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id >= 0:
                self.arm_joint_ids.append(joint_id)
            else:
                print(f"⚠️  Joint '{name}' not found")
        
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
        if site_id >= 0:
            return self.data.site_xpos[site_id].copy()
        else:
            # Fallback to last link position
            return self.data.xpos[-1].copy()

    def apply_joint_velocity(self, joint_id, velocity):
        """Apply velocity command directly to a joint"""
        if joint_id >= 0 and joint_id < len(self.data.ctrl):
            self.data.ctrl[joint_id] = velocity

    def apply_gripper_control(self, target_position):
        """Control Robotiq 2F85 gripper using tendon actuator (0 = open, 255 = closed)"""
        if self.gripper_actuator_id >= 0:
            self.data.ctrl[self.gripper_actuator_id] = target_position

    # =================================================================
    # EXERCISE: IMPLEMENT YOUR POSITION CONTROLLER HERE
    # =================================================================
    
    def compute_jacobian(self):
        """
        Compute the Jacobian matrix for the robot arm.
        
        Returns:
            J (numpy.ndarray): 6x7 Jacobian matrix (3 position + 3 rotation) mapping joint velocities to end-effector velocity
        """
        # Get end effector site ID
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripper_center")
        
        if site_id >= 0:
            # Initialize Jacobian matrices
            jacp = np.zeros((3, self.model.nv))  # Position Jacobian
            jacr = np.zeros((3, self.model.nv))  # Rotation Jacobian
            #print(self.data.qpos)
            # Compute Jacobian using MuJoCo
            #mujoco.mj_jac(self.model, self.data, jacp, jacr, self.data.site_xpos[site_id], site_id)
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, site_id)
            #print(f"🔍 Jacobian Position Part:\n{jacp}")
            # Combine position and rotation Jacobians, return only arm joints (first 7 DOF)
            J_combined = np.concatenate((jacp[:, :7], jacr[:, :7]), axis=0)
            return J_combined
        else:
            return np.zeros((6, 7))
    
    def position_controller(self):
        """
        EXERCISE: Implement a position controller to move the end effector to self.target_position
        
        Available variables:
        - self.target_position: Target position [x, y, z] (numpy array)
        - self.kp_position: Proportional gain
        - self.kd_position: Derivative gain  
        - self.max_joint_velocity: Maximum allowed velocity per joint (rad/s)
        - self.previous_position_error: Previous error for derivative term
        - self.model: MuJoCo model
        - self.data: MuJoCo data
        - self.arm_joint_ids: List of arm joint IDs
        
        Available methods:
        - self.get_end_effector_position(): Returns current end effector position
        - self.compute_jacobian(): Returns 3x7 Jacobian matrix
        - self.apply_joint_velocity(joint_id, velocity): Apply velocity command to a joint
        
        TODO: Implement your controller here!
        Steps:
        1. Get current end effector position
        2. Calculate position error
        3. Calculate derivative of error  
        4. Compute desired end effector velocity/acceleration
        5. Use Jacobian to map to joint space
        6. Apply joint velocity commands
        
        Suggested approaches:
        - PID controller in Cartesian space
        - Jacobian transpose method
        - Jacobian pseudoinverse method
        - Operational space control
        """
        
        if not self.controller_enabled:
            return
            
        # ============================================
        # YOUR CONTROLLER IMPLEMENTATION
        # ============================================
        
        # 1. Get current position and orientation
        current_pos = self.get_end_effector_position()
        
        # Get current end effector orientation as rotation matrix
        orient_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, "gripper_center")
        current_orient_mat = self.data.site_xmat[orient_id].reshape(3,3)

        # Get target position from door handle site (consistent approach)
        handle_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, 'door_handle_site')
        if handle_site_id >= 0:
            self.target_position = self.data.site_xpos[handle_site_id].copy()
        else:
            # Fallback if site not found
            self.target_position = np.array([0.5, 0.2, 0.4])  # [x, y, z] in meters
        #print(f"Goal position for end effector: {self.target_position}")
        
        # Get target orientation from box using consistent quaternion approach
        box_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "box_door_assembly")
        if box_id >= 0:
            # Use the body's rotation matrix directly for consistency
            target_orient_mat = self.data.xmat[box_id].copy().reshape(3, 3)
            
            # Apply desired orientation transformation (90 degree rotation about Y axis)
            rot_y_90 = np.array([
                [0, 0, 1],   # X' = Z
                [0, 1, 0],   # Y' = Y  
                [-1, 0, 0]   # Z' = -X
            ])
            target_orient_mat = target_orient_mat @ rot_y_90
        else:
            target_orient_mat = np.eye(3)  # Identity matrix as fallback

        # Get current joint velocities for the arm (first 7 joints)
        current_joint_vel = self.data.qvel[:7]  # First 7 DOF are arm joints
        
        # Get current angular velocity of end effector in world frame
        # Use the rotational Jacobian to get angular velocity
        J_full = self.compute_jacobian()  # This gives us 6x7 jacobian

        J_rot = J_full[3:6, :7]  # Rotational part of Jacobian
        current_angular_vel = J_rot @ current_joint_vel  # Angular velocity in world frame
        

        # 2. Calculate position error
        position_error = self.target_position - current_pos
        
        #print(f"Target Orient:\n{target_orient_mat}\nCurrent Orient:\n{current_orient_mat}")
        # 3. Calculate orientation error using rotation matrices (fixed direction)
        # Orientation error: R_error = R_current^T * R_target (corrected order)
        R_error = current_orient_mat.T @ target_orient_mat
        
        # Convert rotation matrix error to axis-angle representation
        # For small angles, the skew-symmetric part gives us the orientation error vector
        orientation_error = np.array([
            R_error[2, 1] - R_error[1, 2],  # rotation about x-axis
            R_error[0, 2] - R_error[2, 0],  # rotation about y-axis  
            R_error[1, 0] - R_error[0, 1]   # rotation about z-axis
        ]) * 0.5
        
        # 4. Calculate angular velocity error
        # Target is stationary, so desired angular velocity is zero
        #target_angular_vel = np.zeros(3)  # Desired angular velocity is zero for stationary target
        target_angular_vel= self.data.qvel[3:6]  # Target angular velocity from free joint (door assembly)
        angular_vel_error = target_angular_vel - current_angular_vel
        
        # 5. Get Jacobian (position part only, 3x7)
        J = self.compute_jacobian()
        #print(f"🔍 Jacobian:\n{J}")
        J_pos = J[:3, :7]  # Take only position Jacobian for arm joints
        
        # 6. Controller gains (velocity control - enhanced Z control and proper orientation)
        K_pos = np.diag([1.5, 1.5, 2.0]) *0.5         # Position velocity gains - higher Z gain for better vertical control
        K_orient = np.diag([0.8, 0.8, 0.8])*0.0001       # Orientation velocity gain matrix
        K_angular_vel = np.diag([0.5, 0.5, 0.5]) *0.0001    # Angular velocity damping gain matrix

        # 7. Compute desired Cartesian velocities (PD control in Cartesian space)
        desired_position_velocity = K_pos @ position_error
        #desired_angular_velocity = K_orient @ orientation_error + K_angular_vel @ angular_vel_error
        desired_angular_velocity = K_angular_vel @ angular_vel_error
        # Combine position and angular velocities for 6DOF control
        #desired_cartesian_velocity = np.concatenate([desired_position_velocity, desired_angular_velocity])
        desired_cartesian_velocity = np.concatenate([desired_position_velocity,np.zeros(3)])  # Only position control for simplicity

        # 7. Map to joint space using redundant manipulator control with null space projection
        try:
            # Use full 6DOF Jacobian for position and orientation control
            J_full = self.compute_jacobian()  # 6x7 Jacobian (position + rotation)
            
            # Add damping to avoid singularities (Damped Least Squares)
            lambda_damping = 0.0001  # Damping factor
            J_damped = J_full @ J_full.T + lambda_damping**2 * np.eye(6)
            J_pinv = J_full.T @ np.linalg.inv(J_damped)
            #J_pinv = np.linalg.pinv(J_full)
            
            # Primary task: Cartesian space velocity control
            joint_velocities_primary = J_pinv @ desired_cartesian_velocity
            
            # Null space projection for redundant manipulator
            # (I - J# * J) projects into the null space of the Jacobian
            null_space_proj = np.eye(7) - J_pinv @ J_full
            
            # Secondary task: Joint velocity damping to avoid singularities and improve stability
            K_null = 0.1  # Null space gain for velocity control
            joint_damping_velocities = -K_null * current_joint_vel  # Damp joint velocities
            
            # Combined velocities: primary task + null space damping
            #joint_velocities = joint_velocities_primary + null_space_proj @ joint_damping_velocities
            joint_velocities = joint_velocities_primary

            #print(f"🎯 Pos Error: {position_error}, Orient Error: {orientation_error}, Velocities: {joint_velocities}")
            
        except np.linalg.LinAlgError:
            print("⚠️ Jacobian computation failed, using zero velocities")
            joint_velocities = np.zeros(7)
        
        # 8. Apply velocity commands with limits
        max_velocity = 2.0  # Maximum velocity limit (rad/s)
        joint_velocities_clipped = np.clip(joint_velocities, -max_velocity, max_velocity)
        
        # Set velocity commands directly to ctrl (velocity control)
        self.data.ctrl[:len(joint_velocities_clipped)] = joint_velocities_clipped
        if self.model.nu > len(joint_velocities_clipped):
            self.data.ctrl[len(joint_velocities_clipped):] = 0
        
        # 9. Store target orientation for frame visualization
        self.current_target_orientation = target_orient_mat.copy()
        
        # 10. Collect data for plotting
        self.collect_controller_data(current_pos, current_orient_mat, target_orient_mat, 
                                   position_error, orientation_error, current_joint_vel, current_angular_vel)
        
        # 11. Update previous error for next iteration
        self.previous_position_error = position_error.copy()
    
    def update_target_marker(self):
        """Update the visual target marker position"""
        target_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "target_marker")
        if target_body_id >= 0:
            # Update target marker position
            self.data.xpos[target_body_id] = self.target_position

    def update_target_frame(self, target_orient_mat):
        """Update the 3D frame marker for target position and orientation"""
        target_frame_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "Base_Box")
        if target_frame_id >= 0:
            # Update target frame position
            self.data.xpos[target_frame_id] = self.target_position
            # Update target frame orientation (rotation matrix to quaternion)
            target_quat = np.zeros(4)  # [w, x, y, z] format
            mujoco.mju_mat2Quat(target_quat, target_orient_mat.flatten())
            self.data.xquat[target_frame_id] = target_quat
            # Apply forward kinematics to update the visualization
            mujoco.mj_forward(self.model, self.data)

    def apply_impulse_to_door(self):
        """Apply impulse to the door"""
        if self.door_joint_id >= 0:
            # Find the DOF index for the door hinge
            dof_adr = self.model.jnt_dofadr[self.door_joint_id]
            self.data.qvel[dof_adr] += self.impulse_strength
            print("💥 Applied impulse to door")

    def apply_impulse_to_assembly(self):
        """Apply impulse to the box/door assembly"""
        if self.assembly_freejoint_id >= 0:
            # Apply random impulses in X, Y, Z directions
            import random
            dof_adr = self.model.jnt_dofadr[self.assembly_freejoint_id]
            # Linear impulses (first 3 DOF of freejoint)
            self.data.qvel[dof_adr] += (random.random() - 0.5) * self.impulse_strength  # Linear X velocity
            self.data.qvel[dof_adr + 1] += (random.random() - 0.5) * self.impulse_strength  # Linear Y velocity  
            self.data.qvel[dof_adr + 2] += (random.random() - 0.5) * self.impulse_strength  # Linear Z velocity
            # Angular impulses (last 3 DOF of freejoint)
            self.data.qvel[dof_adr + 4] += (random.random() - 0.5) * self.impulse_strength * 0.5  # Angular X
            self.data.qvel[dof_adr + 5] += (random.random() - 0.5) * self.impulse_strength * 0.5  # Angular Y
            self.data.qvel[dof_adr + 6] += (random.random() - 0.5) * self.impulse_strength * 0.5  # Angular Z
            print("💥 Applied random impulse to box assembly")

    def reset_to_home(self):
        """Reset simulation to initial state"""
        # Reset to home keyframe if available
        home_key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, "home")
        if home_key_id >= 0:
            mujoco.mj_resetDataKeyframe(self.model, self.data, home_key_id)
        
        # Clear all velocities and forces
        self.data.qvel[:] = 0
        self.data.qfrc_applied[:] = 0
        
        print("🏠 Reset to home - free physics mode")

    def collect_controller_data(self, current_pos, current_orient_mat, target_orient_mat, 
                              position_error, orientation_error, current_joint_vel, current_angular_vel):
        """Collect data for plotting"""
        if self.simulation_start_time is None:
            self.simulation_start_time = time.time()
        
        current_time = time.time() - self.simulation_start_time
        
        # Store data
        self.time_data.append(current_time)
        
        # Store position and orientation errors as components (not just magnitude)
        self.position_error_data.append(position_error.copy())  # 3D position error components
        self.orientation_error_data.append(orientation_error.copy())  # 3D orientation error components
        self.current_position_data.append(current_pos.copy())
        self.target_position_data.append(self.target_position.copy())  # Store target position
        
        # Use quaternions instead of Euler angles to avoid gimbal lock and discontinuities
        from scipy.spatial.transform import Rotation
        
        # Convert rotation matrices to quaternions (continuous representation)
        current_rotation = Rotation.from_matrix(current_orient_mat)
        current_quat = current_rotation.as_quat()  # [x, y, z, w] format
        
        target_rotation = Rotation.from_matrix(target_orient_mat)
        target_quat = target_rotation.as_quat()  # [x, y, z, w] format
        
        # Ensure quaternion continuity by checking sign flip
        if len(self.current_orientation_data) > 0:
            prev_current_quat = np.array(self.current_orientation_data[-1])
            prev_target_quat = np.array(self.target_orientation_data[-1])
            
            # If quaternion dot product is negative, flip sign to maintain continuity
            if np.dot(current_quat, prev_current_quat) < 0:
                current_quat = -current_quat
            if np.dot(target_quat, prev_target_quat) < 0:
                target_quat = -target_quat
        
        self.current_orientation_data.append(current_quat)
        self.target_orientation_data.append(target_quat)
        
        # Store joint data
        current_joints = self.data.qpos[:7].copy()
    
        # Store joint data
        self.joint_values_data.append(current_joints)
        self.joint_velocities_data.append(current_joint_vel.copy())
        
        # Calculate end effector linear velocity using Jacobian
        J_full = self.compute_jacobian()
        J_pos = J_full[:3, :7]  # Position Jacobian
        current_linear_vel = J_pos @ current_joint_vel
        self.linear_velocity_data.append(current_linear_vel.copy())
        self.angular_velocity_data.append(current_angular_vel.copy())

    def plot_controller_data(self):
        """Plot comprehensive controller data analysis"""
        if len(self.time_data) < 10:  # Need some data to plot
            print("⚠️ Not enough data to plot (need at least 10 data points)")
            return
            
        # Convert deques to numpy arrays for plotting with error checking
        try:
            time_array = np.array(self.time_data)
            pos_error_array = np.array(self.position_error_data)  # Shape: (n, 3)
            orient_error_array = np.array(self.orientation_error_data)  # Shape: (n, 3)
            position_array = np.array(self.current_position_data)  # Shape: (n, 3)
            target_position_array = np.array(self.target_position_data)  # Shape: (n, 3)
            current_orientation_array = np.array(self.current_orientation_data)  # Shape: (n, 4) - quaternions
            target_orientation_array = np.array(self.target_orientation_data)  # Shape: (n, 4) - quaternions
            
            # Handle joint data with error checking
            if len(self.joint_values_data) > 0 and len(self.joint_values_data[0]) > 0:
                joint_array = np.array(self.joint_values_data)  # Shape: (n, 7)
            else:
                joint_array = np.zeros((len(time_array), 7))
                
            if len(self.joint_velocities_data) > 0 and len(self.joint_velocities_data[0]) > 0:
                joint_vel_array = np.array(self.joint_velocities_data)  # Shape: (n, 7)
            else:
                joint_vel_array = np.zeros((len(time_array), 7))
                
            linear_vel_array = np.array(self.linear_velocity_data)  # Shape: (n, 3)
            angular_vel_array = np.array(self.angular_velocity_data)  # Shape: (n, 3)
            
        except Exception as e:
            print(f"❌ Error processing data arrays: {e}")
            return
        
        # Create comprehensive plots with more subplots
        fig, axes = plt.subplots(3, 3, figsize=(20, 18))
        fig.suptitle('Advanced Controller Performance Analysis', fontsize=18, fontweight='bold')
        
        # Plot 1: Position Error Components
        axes[0, 0].plot(time_array, pos_error_array[:, 0], 'r-', linewidth=2, label='Error X')
        axes[0, 0].plot(time_array, pos_error_array[:, 1], 'g-', linewidth=2, label='Error Y')
        axes[0, 0].plot(time_array, pos_error_array[:, 2], 'b-', linewidth=2, label='Error Z')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Position Error (m)')
        axes[0, 0].set_title('Position Error Components')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Plot 2: Orientation Error Components
        axes[0, 1].plot(time_array, orient_error_array[:, 0], 'r-', linewidth=2, label='Error Rx')
        axes[0, 1].plot(time_array, orient_error_array[:, 1], 'g-', linewidth=2, label='Error Ry')
        axes[0, 1].plot(time_array, orient_error_array[:, 2], 'b-', linewidth=2, label='Error Rz')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Orientation Error (rad)')
        axes[0, 1].set_title('Orientation Error Components')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Plot 3: All Joint Values in One Graph
        colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink']
        for i in range(min(7, joint_array.shape[1])):
            axes[0, 2].plot(time_array, joint_array[:, i], color=colors[i], linewidth=2, label=f'Joint {i+1}')
        axes[0, 2].set_xlabel('Time (s)')
        axes[0, 2].set_ylabel('Joint Angle (rad)')
        axes[0, 2].set_title('All Joint Values')
        axes[0, 2].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 2].grid(True)
        
        # Plot 4: Current Position vs Target
        axes[1, 0].plot(time_array, position_array[:, 0], 'r-', linewidth=2, label='Current X')
        axes[1, 0].plot(time_array, position_array[:, 1], 'g-', linewidth=2, label='Current Y')
        axes[1, 0].plot(time_array, position_array[:, 2], 'b-', linewidth=2, label='Current Z')
        # Add target position as time-varying lines
        axes[1, 0].plot(time_array, target_position_array[:, 0], 'r--', linewidth=2, alpha=0.7, label='Target X')
        axes[1, 0].plot(time_array, target_position_array[:, 1], 'g--', linewidth=2, alpha=0.7, label='Target Y')
        axes[1, 0].plot(time_array, target_position_array[:, 2], 'b--', linewidth=2, alpha=0.7, label='Target Z')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Position (m)')
        axes[1, 0].set_title('End Effector Position vs Target')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Plot 5: Current vs Target Orientation (Quaternions - Continuous)
        axes[1, 1].plot(time_array, current_orientation_array[:, 0], 'r-', linewidth=2, label='Current qx')
        axes[1, 1].plot(time_array, current_orientation_array[:, 1], 'g-', linewidth=2, label='Current qy')
        axes[1, 1].plot(time_array, current_orientation_array[:, 2], 'b-', linewidth=2, label='Current qz')
        axes[1, 1].plot(time_array, current_orientation_array[:, 3], 'm-', linewidth=2, label='Current qw')
        axes[1, 1].plot(time_array, target_orientation_array[:, 0], 'r--', linewidth=1, alpha=0.7, label='Target qx')
        axes[1, 1].plot(time_array, target_orientation_array[:, 1], 'g--', linewidth=1, alpha=0.7, label='Target qy')
        axes[1, 1].plot(time_array, target_orientation_array[:, 2], 'b--', linewidth=1, alpha=0.7, label='Target qz')
        axes[1, 1].plot(time_array, target_orientation_array[:, 3], 'm--', linewidth=1, alpha=0.7, label='Target qw')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Quaternion Components')
        axes[1, 1].set_title('Orientation: Current vs Target (Quaternions)')
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
        axes[1, 1].grid(True)
        
        # Plot 6: End Effector Linear Velocities
        axes[1, 2].plot(time_array, linear_vel_array[:, 0], 'r-', linewidth=2, label='Vel X')
        axes[1, 2].plot(time_array, linear_vel_array[:, 1], 'g-', linewidth=2, label='Vel Y')
        axes[1, 2].plot(time_array, linear_vel_array[:, 2], 'b-', linewidth=2, label='Vel Z')
        axes[1, 2].set_xlabel('Time (s)')
        axes[1, 2].set_ylabel('Linear Velocity (m/s)')
        axes[1, 2].set_title('End Effector Linear Velocities')
        axes[1, 2].legend()
        axes[1, 2].grid(True)
        
        # Plot 7: End Effector Angular Velocities
        axes[2, 0].plot(time_array, angular_vel_array[:, 0], 'r-', linewidth=2, label='ωx')
        axes[2, 0].plot(time_array, angular_vel_array[:, 1], 'g-', linewidth=2, label='ωy')
        axes[2, 0].plot(time_array, angular_vel_array[:, 2], 'b-', linewidth=2, label='ωz')
        axes[2, 0].set_xlabel('Time (s)')
        axes[2, 0].set_ylabel('Angular Velocity (rad/s)')
        axes[2, 0].set_title('End Effector Angular Velocities')
        axes[2, 0].legend()
        axes[2, 0].grid(True)
        
        # Plot 8: Joint Velocities (first 4 joints)
        for i in range(min(4, joint_vel_array.shape[1])):
            axes[2, 1].plot(time_array, joint_vel_array[:, i], color=colors[i], linewidth=2, label=f'Joint {i+1}')
        axes[2, 1].set_xlabel('Time (s)')
        axes[2, 1].set_ylabel('Joint Velocity (rad/s)')
        axes[2, 1].set_title('Joint Velocities (Joints 1-4)')
        axes[2, 1].legend()
        axes[2, 1].grid(True)
        
        # Plot 9: Joint Velocities (last 3 joints)
        for i in range(4, min(7, joint_vel_array.shape[1])):
            axes[2, 2].plot(time_array, joint_vel_array[:, i], color=colors[i], linewidth=2, label=f'Joint {i+1}')
        axes[2, 2].set_xlabel('Time (s)')
        axes[2, 2].set_ylabel('Joint Velocity (rad/s)')
        axes[2, 2].set_title('Joint Velocities (Joints 5-7)')
        axes[2, 2].legend()
        axes[2, 2].grid(True)
        
        plt.tight_layout()
        
        # Save the plot to file
        plt.savefig('controller_performance_analysis.png', dpi=300, bbox_inches='tight')
        print("📊 Advanced controller data saved to 'controller_performance_analysis.png'!")
        plt.show()
        # Also show the plot if running interactively
        try:
            plt.show(block=True)  # Non-blocking show
        except:
            pass  # If display not available, just save the file

    def run_simulation(self, duration=None):
        """Run the combined simulation
        Args:
            duration: Optional duration in seconds to run simulation. If None, runs indefinitely.
        """
        
        # Initialize to home position
        #self.reset_to_home()
        
        simulation_start = time.time()
        
        with mujoco.viewer.launch_passive(self.model, self.data) as viewer:
            last_print_time = 0
            
            while viewer.is_running():
                # Check duration limit
                if duration is not None and (time.time() - simulation_start) > duration:
                    break
                    
                step_start = time.time()
                
                # Clear any previous forces
                self.data.qfrc_applied[:] = 0
                
                # Run position controller (if enabled)
                self.position_controller()
                
                # Update visual target marker
                #self.update_target_marker()
                
                # Update 3D target frame with orientation
                if hasattr(self, 'current_target_orientation'):
                    self.update_target_frame(self.current_target_orientation)
                
                # Step simulation
                mujoco.mj_step(self.model, self.data)
                
                # Update viewer
                viewer.sync()
                
                # Print status occasionally
                if time.time() - last_print_time > 2.0:
                    current_pos = self.get_end_effector_position()
                    door_dof_adr = self.model.jnt_dofadr[self.door_joint_id] if self.door_joint_id >= 0 else -1
                    door_angle = self.data.qpos[door_dof_adr] * 180 / np.pi if door_dof_adr >= 0 else 0
                    
                    # Get gripper control value
                    gripper_ctrl = self.data.ctrl[self.gripper_actuator_id] if self.gripper_actuator_id >= 0 else 0
                    
                    # Calculate position error
                    pos_error = np.linalg.norm(self.target_position - current_pos)
                    controller_status = "ON" if self.controller_enabled else "OFF"
                    
                    print(f"End-Effector: [{current_pos[0]:.2f}, {current_pos[1]:.2f}, {current_pos[2]:.2f}] | "
                          f"Target: [{self.target_position[0]:.2f}, {self.target_position[1]:.2f}, {self.target_position[2]:.2f}] | "
                          f"Error: {pos_error:.3f}m | Controller: {controller_status} | Gripper: {gripper_ctrl:.0f}/255")
                    last_print_time = time.time()
                
                # Timing
                elapsed = time.time() - step_start
                sleep_time = self.model.opt.timestep - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        
        print("👋 Simulation ended")
        
        # Plot the collected data
        #self.plot_controller_data()


def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run the combined zero-gravity simulation')
    parser.add_argument('--duration', '-d', type=float, default=None, 
                        help='Duration to run simulation in seconds (default: run indefinitely)')
    parser.add_argument('--plot-only', action='store_true',
                        help='Skip simulation, just generate plots from existing data')
    
    args = parser.parse_args()
    
    simulation = None
    try:
        simulation = CombinedZeroGravitySimulation()
        
        if not args.plot_only:
            simulation.run_simulation(duration=args.duration)
        else:
            print("📊 Plot-only mode: generating plots...")
            
    except KeyboardInterrupt:
        print("\n👋 Simulation interrupted by user")
    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        # Always try to plot data if simulation was created and has data
        if simulation is not None and len(simulation.time_data) > 10:
            print("📊 Generating plots from collected data...")
            simulation.plot_controller_data()
        elif simulation is not None:
            print("⚠️ Not enough data collected for plotting (need at least 10 data points)")


if __name__ == "__main__":
    main()
