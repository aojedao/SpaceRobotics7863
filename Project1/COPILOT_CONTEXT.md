# Wall Crawler MuJoCo Simulation - Copilot Context

## Project Overview
Dual-arm wall-crawler robot simulation for ISS module traversal using MuJoCo physics engine.
The robot alternates between two KUKA iiwa14 arms with Robotiq 2F85 grippers to traverse walls.

## Current State (December 2, 2025)

### ⚠️ KNOWN ISSUES - NEED FIXING

1. **Trajectories Not Completing Successfully**
   - Robot often fails to reach waypoints within thresholds
   - Success rate is low (~50% or less)
   - IK solver may be getting stuck in local minima

2. **Recovery System Not Working Properly**
   - Current implementation: release control for 10 seconds, retry up to 10 times
   - The "release control" approach is NOT effective
   - Simply setting `ctrl[:] = 0` doesn't help the robot recover
   - Need a better recovery strategy (e.g., random perturbation, backtracking, different IK seed)

3. **Stuck Detection Triggers But Recovery Fails**
   - 30-second timeout detects stuck conditions correctly
   - But the recovery action (releasing control) doesn't help robot escape local minima
   - Robot just drifts or stays stuck after release

### What's Implemented (Partially Working)

1. **Random Goal Generation**: Goals randomly selected from 5 walls
   - Front, Back, Left, Right, and Ceiling walls supported
   - Uses ISS_MODULE bounds for valid positions

2. **Complete Route Visualization**: All waypoints shown with color coding
   - 🟢 Green spheres: Completed waypoints
   - 🟡 Yellow sphere: Current target waypoint
   - 🔵 Blue spheres: Upcoming waypoints
   - Blue lines connect waypoints showing planned path

3. **Trajectory Error Plotting**: Now generates plots correctly
   - Fixed: Trajectory plot was nested inside anchor deviation block
   - Now plots independently to `trajectory_error_analysis.png`

4. **Screw Task at Goal** (when trajectory completes):
   - Screw model spawns at final goal position
   - Active arm's joint7 rotates 3 full turns at 0.5 rad/s

5. **Torque Balancing**:
   - Free arm applies 30% counter-torque on joint7 during screw task

6. **Per-Joint Gain Scaling**:
   - Joints 1-2: 1.8x multiplier (stronger for base positioning)
   - Joints 3-4: 1.0x (standard)
   - Joints 5-6: 0.8x (reduced for wrist)
   - Joint 7: 0.6x (lowest for end-effector precision)

### Technical Configuration

#### IK Controller Gains
```python
kp_position = 500.0          # Position gain
kd_position = 15.0           # Damping gain  
lambda_dls = 0.012           # Damped least squares regularization
joint_gain_scaling = [1.8, 1.8, 1.0, 1.0, 0.8, 0.8, 0.6]  # Per-joint multipliers
```

#### Success Thresholds
```python
final_threshold = 0.12       # 12cm for final waypoint
intermediate_threshold = 0.18 # 18cm for intermediate waypoints
forced_progress_threshold = 0.50  # 50cm emergency threshold
```

#### Stuck Detection Parameters (NEEDS IMPROVEMENT)
```python
stuck_timeout = 30.0         # Seconds without progress - works
max_recovery_attempts = 10   # Maximum retry count
release_duration = 10.0      # Seconds to release control - NOT EFFECTIVE
```

### Key Files

#### wall_crawler_mujoco.py (~3061 lines)
Main simulation file containing:
- `WallCrawlerController` class with state machine
- `apply_arm_control()`: Per-joint gain scaling
- `_render_visualization_geoms()`: Route visualization
- Stuck detection in `reaching_waypoint` phase
- Screw spawning and rotation in `trajectory_complete` phase

#### dual_arm_robot.xml
MuJoCo model with:
- Two KUKA iiwa14 arms
- Robotiq 2F85 grippers
- Screw body (hidden at pos="100 100 100" until spawned)

### State Machine Phases
1. `reaching_anchor`: Moving arm approaches current waypoint
2. `reaching_waypoint`: Same as above, with stuck detection
3. `trajectory_complete`: Screw spawning and rotation task

### Running the Simulation
```bash
cd /home/aojedao/Documents/NYU/SpaceRobotics/SpaceRobotics7863/Project1
/home/aojedao/miniconda3/bin/conda run -n space-robotics --no-capture-output python wall_crawler_mujoco.py
```

### TODO - Priority Fixes Needed

1. **Fix Recovery Strategy** - The current "release control" approach doesn't work
   - Options to try:
     - Random joint perturbation to escape local minima
     - Backtrack to previous waypoint and retry
     - Use different IK seed/configuration
     - Apply small random torques instead of zero
     - Temporarily increase gains to "push through"

2. **Improve Trajectory Success Rate**
   - May need to adjust thresholds
   - Consider adaptive thresholds based on distance
   - Better path planning to avoid unreachable configurations

3. **Debug Why IK Gets Stuck**
   - Log joint configurations when stuck
   - Visualize Jacobian condition number
   - Check for singularities

### ISS Module Bounds
```python
ISS_MODULE = {
    'x_min': -2.1, 'x_max': 3.9,  # 6m length
    'y_min': -0.5, 'y_max': 1.7,  # 2.2m width  
    'z_min': 0.1,  'z_max': 2.2   # 2.1m height
}
```

### Generated Plots
1. `anchor_deviation_analysis.png` - Anchor stability per waypoint
2. `trajectory_error_analysis.png` - Tracking error with initial vs final bars (NOW WORKING)
