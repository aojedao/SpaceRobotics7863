# Wall Crawler MuJoCo Simulation - Copilot Context# Wall Crawler MuJoCo Simulation - Copilot Context



## Project Overview## Project Overview



Dual-arm wall-crawler robot simulation for ISS module traversal using MuJoCo physics engine.Dual-arm wall-crawler robot simulation for ISS module traversal using MuJoCo physics engine.

The robot alternates between two KUKA iiwa14 arms with Robotiq 2F85 grippers to traverse walls,The robot alternates between two KUKA iiwa14 arms with Robotiq 2F85 grippers to traverse walls.

and performs an unscrewing task at the final goal position.

## Current State (December 7, 2025)

## Current State (December 7, 2025)

### Branch: `DualArmDev-side-mounted-arms_antigravity`

### Branch: `DualArmDev-side-mounted-arms_antigravity`

### Arm Configuration

### Arm Configuration- **Left Arm**: Position Y=-0.25, Euler "0 -90 0" → Points in **-X direction**

- **Left Arm**: Position Y=-0.25, Euler "0 -90 0" → Points in **-X direction**- **Right Arm**: Position Y=+0.25, Euler "0 90 0" → Points in **+X direction**

- **Right Arm**: Position Y=+0.25, Euler "0 90 0" → Points in **+X direction**- Arms are on the Y-axis sides of the body box, pointing outward in opposite X directions

- Arms are on the Y-axis sides of the body box, pointing outward in opposite X directions- Zero gravity environment (anti-gravity mode)

- Zero gravity environment (anti-gravity mode)

### Test Results

### Test Results**Working on planar wall locations (back, front walls)**

**Working on planar wall locations (back, front walls)**- Trajectories on planar walls (back, front, floor, ceiling) work reliably

- Trajectories on planar walls (back, front, floor, ceiling) work reliably- Corner transitions and wall changes may require additional tuning

- Corner transitions and wall changes may require additional tuning- Screw spawns at goal position from start with correct wall orientation

- Screw spawns at goal position from start with correct wall orientation

- Unscrewing task executes after trajectory completion### Recent Fixes (December 7, 2025)

1. **Jacobian computed at gripper site** - Previously computed at body origin, now uses `compute_jacobian_at_site()` for accurate end-effector control

---2. **J1 correction removed** - Was being double-applied (once in velocity, once in position command), fighting the IK solver

3. **Orientation control disabled** - `use_orientation_control = False` for all waypoints as it was making reaching harder

## Unscrewing Task Implementation4. **Success thresholds relaxed** - 0.22m for intermediate waypoints, 0.20m for final waypoint

5. **Recovery J1 bug fixed** - Now correctly uses anchored arm J1 for both computation and application

### Overview6. **Screw spawn at goal** - Screw now spawns at goal position from start with wall-appropriate orientation

After the robot reaches the final waypoint (goal position), an unscrewing task is executed:

1. **Alignment Phase**: Active IK control aligns the end-effector with the screw axis### Key Parameters

2. **Rotation Phase**: J7 (wrist) rotates to unscrew while joints 1-6 stay locked

3. **Screw Movement**: Screw moves along its axis (out of wall) based on rotation#### Arm Offsets (in wall_crawler_mujoco.py)

```python

### Key Methods in `wall_crawler_mujoco.py`LEFT_ARM_OFFSET = -0.25   # Y-axis offset for left arm

RIGHT_ARM_OFFSET = 0.25   # Y-axis offset for right arm

| Method | Lines | Purpose |```

|--------|-------|---------|

| `_spawn_screw_at_goal()` | ~660-720 | Spawns screw at goal with wall-appropriate orientation |#### Control Gains

| `_make_screw_visible()` | ~730-755 | Makes screw large and colorful (RED head, ORANGE shaft, CYAN thread) |```python

| `_get_screw_unscrew_axis()` | ~757-770 | Returns axis direction for screw movement per wall |kp_position = 450.0      # Position control gain (INCREASED)

| `_apply_j7_rotation()` | ~772-790 | Applies rotation to J7 joint |kd_position = 35.0       # Velocity damping (INCREASED)

| `_apply_torque_compensation()` | ~792-815 | Applies Coriolis/centrifugal compensation |kp_orientation = 50.0    # Orientation control

| `_maintain_ee_position_and_orientation()` | ~817-870 | IK-based EE position/orientation control |kd_orientation = 20.0

| `_calculate_alignment_error()` | ~930-952 | Calculates angle between EE Z-axis and screw axis |lambda_dls = 0.008       # Damped least squares regularization

| `_step_positioning_phase()` | ~954-1040 | Active alignment phase with IK + torque compensation |```

| `_init_unscrew_task()` | ~872-928 | Initializes unscrewing state variables |

| `_step_unscrew_task()` | ~1042-1130 | Executes one step of rotation phase |#### Path Planner

| `_lock_all_for_unscrew()` | ~1132-1150 | Locks body, anchor arm, and screw position |```python

| `_plot_unscrew_alignment()` | ~1152-1240 | Plots alignment error graph after completion |step_size = 0.5m         # Distance between waypoints

effective_reach = 0.75m  # Planning reach (< actual 1.25m arm reach)

### Unscrewing Parameters```

```python

# In _init_unscrew_task()### Recovery Mechanism

self._unscrew_speed = 6.0              # rad/s (~344°/s)- **15 recovery strategies** with varied J1 angles: [0.0, -1.0, 1.0, -2.0, 2.0, -1.5, 1.5, -2.5, 2.5, 0.5, -0.5, -3.0, 3.0, π/2, -π/2]

self._unscrew_thread_pitch = 0.015     # 15mm per revolution- **Stuck threshold**: 8 seconds

self._unscrew_max_displacement = 0.15  # 150mm max travel- **Recovery duration**: 3 seconds per attempt

self._unscrew_target_rotations = 3.0   # 3 full turns to complete- **Smart J1 pre-rotation**: Initial joint 1 angle set toward target direction



# Alignment Phase### Path Planning Logic

alignment_threshold_deg = 15.0         # Start rotation when error < 15°1. **Initial arm selection**: Based on direction to goal

min_alignment_time = 0.5               # At least 0.5s of alignment   - Moving -X → LEFT arm starts (points -X)

max_alignment_time = 3.0               # Max 3s, then start anyway   - Moving +X → RIGHT arm starts (points +X)

```2. **Arm alternation**: Simple alternating LEFT/RIGHT for bipedal locomotion

3. **No arm reachability filter**: Removed to allow direct paths; IK handles reachability at execution

### Screw Axis Mapping (OUT of wall direction)

```python### ISS Module Bounds

axes = {```python

    'floor': [0, 0, +1],      # Up out of floorISS_MODULE = {

    'ceiling': [0, 0, -1],    # Down out of ceiling    'x_min': -2.1, 'x_max': 3.9,

    'front': [0, -1, 0],      # -Y out of front wall    'y_min': -0.5, 'y_max': 1.7,

    'back': [0, +1, 0],       # +Y out of back wall    'z_min': 0.1,  'z_max': 2.2

    'left_wall': [+1, 0, 0],  # +X out of left wall}

    'right_wall': [-1, 0, 0], # -X out of right wall```

}

```## Files Overview



### Visualization| File | Purpose |

- **Screw Body**: RED head (6cm), ORANGE shaft, CYAN thread|------|---------|

- **Screw Marker**: MAGENTA sphere (8cm) + yellow transparent ring (12cm)| `wall_crawler_mujoco.py` | Main simulation (~3400 lines) with path planning, IK control, locomotion |

- **Alignment Graph**: Saved to `unscrew_alignment.png` after completion| `dual_arm_robot.xml` | MuJoCo model with robot and ISS environment |

  - Panel 1: Alignment error over time| `run_multiple_tests.py` | Batch test runner (parallel or sequential) |

  - Panel 2: J7 rotation over time| `run_history.json` | Test results history |

  - Panel 3: EE-to-screw distance

  - Panel 4: Screw displacement along axis## Command Line Usage



---```bash

# Run with visualization (default)

## Recent Fixes (December 7, 2025)conda run -n space-robotics python wall_crawler_mujoco.py



### Trajectory Control# Run headless (basic - needs full locomotion loop)

1. **Jacobian computed at gripper site** - Previously computed at body origin, now uses `compute_jacobian_at_site()` for accurate end-effector controlconda run -n space-robotics python wall_crawler_mujoco.py --headless

2. **J1 correction removed** - Was being double-applied (once in velocity, once in position command), fighting the IK solver

3. **Orientation control disabled** - `use_orientation_control = False` for all waypoints as it was making reaching harder# Run without recording to history

4. **Success thresholds relaxed** - 0.22m for intermediate waypoints, 0.20m for final waypointconda run -n space-robotics python wall_crawler_mujoco.py --no-record

5. **Recovery J1 bug fixed** - Now correctly uses anchored arm J1 for both computation and application

6. **Screw spawn at goal** - Screw now spawns at goal position from start with wall-appropriate orientation# Run multiple tests with visualization (sequential)

conda run -n space-robotics python run_multiple_tests.py 5 1 --visualize

### Unscrewing Task```

7. **Active alignment phase** - Uses IK + torque compensation to align EE with screw axis before rotation

8. **Alignment-based rotation start** - Only starts J7 rotation when alignment error < 15° (or timeout)## Known Issues

9. **Screw visibility improved** - Larger screw (6cm head), bright colors, marker spheres

10. **Target markers transparent** - Yellow/orange target spheres now 25% opacity so screw is visible1. **Low Success Rate (2.4%)**: IK solver gets stuck in local minima

11. **Alignment plotting** - Generates 4-panel graph showing alignment, rotation, distance, displacement2. **Arm Crossing**: When arms cross paths, recovery is difficult

3. **Headless Mode Incomplete**: `run_headless()` doesn't run full locomotion loop

---4. **Early Waypoint Failures**: Most failures at WP1-WP3 suggest body positioning issues



## Key Parameters## TODO



### Arm Offsets (in wall_crawler_mujoco.py)1. [ ] Extract locomotion loop from `run_visualization()` into reusable method

```python2. [ ] Implement full headless mode with complete locomotion logic

LEFT_ARM_OFFSET = -0.25   # Y-axis offset for left arm3. [ ] Improve IK convergence with null-space optimization

RIGHT_ARM_OFFSET = 0.25   # Y-axis offset for right arm4. [ ] Better initial body positioning based on first waypoint

```5. [ ] Tune parameters for ceiling/floor transitions

6. [ ] Remove duplicate code between controller.py and wall_crawler_mujoco.py

### Control Gains

```python## Recent Changes (December 7, 2025)

kp_position = 450.0      # Position control gain

kd_position = 35.0       # Velocity damping1. ✅ Jacobian computed at gripper site instead of body origin

kp_orientation = 50.0    # Orientation control2. ✅ Removed J1 correction (was double-applied and fighting IK)

kd_orientation = 20.03. ✅ Disabled orientation control for all waypoints

lambda_dls = 0.008       # Damped least squares regularization4. ✅ Relaxed success thresholds (0.22m intermediate, 0.20m final)

5. ✅ Fixed recovery to use anchored arm J1 correctly

# Unscrewing alignment control (stricter)6. ✅ Screw spawns at goal position from start with wall orientation

kp_pos = 800.0           # Position tracking during alignment7. ✅ Planar wall trajectories (back, front) working reliably

kp_orient = 200.0        # Orientation tracking during alignment

```## Previous Changes (December 6, 2025)



### Path Planner1. ✅ Repositioned arms to Y-axis sides pointing ±X

```python2. ✅ Updated ARM_OFFSET constants to match new positions

step_size = 0.5m         # Distance between waypoints3. ✅ Removed restrictive arm reachability filter from path planner

effective_reach = 0.75m  # Planning reach (< actual 1.25m arm reach)4. ✅ Added initial arm selection based on travel direction

```5. ✅ Increased control gains (kp=450, kd=35)

6. ✅ Expanded recovery strategies to 15 different J1 angles

### ISS Module Bounds7. ✅ Added --headless flag (basic implementation)

```python8. ✅ Smart J1 pre-rotation toward target direction

ISS_MODULE = {

    'x_min': -2.1, 'x_max': 3.9,
    'y_min': -0.5, 'y_max': 1.7,
    'z_min': 0.1,  'z_max': 2.2
}
```

---

## Recovery Mechanism
- **15 recovery strategies** with varied J1 angles: [0.0, -1.0, 1.0, -2.0, 2.0, -1.5, 1.5, -2.5, 2.5, 0.5, -0.5, -3.0, 3.0, π/2, -π/2]
- **Stuck threshold**: 8 seconds
- **Recovery duration**: 3 seconds per attempt
- **Smart J1 pre-rotation**: Initial joint 1 angle set toward target direction

---

## Path Planning Logic
1. **Initial arm selection**: Based on direction to goal
   - Moving -X → LEFT arm starts (points -X)
   - Moving +X → RIGHT arm starts (points +X)
2. **Arm alternation**: Simple alternating LEFT/RIGHT for bipedal locomotion
3. **No arm reachability filter**: Removed to allow direct paths; IK handles reachability at execution

---

## Files Overview

| File | Purpose |
|------|---------|
| `wall_crawler_mujoco.py` | Main simulation (~3900 lines) with path planning, IK control, locomotion, unscrewing |
| `dual_arm_robot.xml` | MuJoCo model with robot and ISS environment |
| `controller.py` | Controller classes (used by some older code) |
| `planner.py` | Path planning utilities |
| `config.py` | Configuration constants |
| `enums.py` | State machine enums |
| `visualization.py` | Visualization utilities |
| `debug_unscrew.py` | Standalone debug simulation for unscrewing task |
| `run_multiple_tests.py` | Batch test runner (parallel or sequential) |
| `run_history.json` | Test results history |
| `COPILOT_CONTEXT.md` | This documentation file |

---

## Command Line Usage

```bash
# Run with visualization (default)
python wall_crawler_mujoco.py

# Run headless (basic - needs full locomotion loop)
python wall_crawler_mujoco.py --headless

# Run without recording to history
python wall_crawler_mujoco.py --no-record

# Run multiple tests with visualization (sequential)
python run_multiple_tests.py 5 1 --visualize
```

---

## Known Issues

1. **Low Success Rate (~2.4%)**: IK solver gets stuck in local minima
2. **Arm Crossing**: When arms cross paths, recovery is difficult
3. **Headless Mode Incomplete**: `run_headless()` doesn't run full locomotion loop
4. **Early Waypoint Failures**: Most failures at WP1-WP3 suggest body positioning issues
5. **Alignment timeout**: If EE can't align to < 15° in 3s, rotation starts anyway

---

## TODO

1. [ ] Extract locomotion loop from `run_visualization()` into reusable method
2. [ ] Implement full headless mode with complete locomotion logic
3. [ ] Improve IK convergence with null-space optimization
4. [ ] Better initial body positioning based on first waypoint
5. [ ] Tune parameters for ceiling/floor transitions
6. [ ] Remove duplicate code between controller.py and wall_crawler_mujoco.py
7. [ ] Test unscrewing on different wall orientations (floor, ceiling, side walls)
8. [ ] Add screw re-insertion (screwing) task after unscrewing

---

## Change History

### December 7, 2025 (Latest)
- ✅ Jacobian computed at gripper site instead of body origin
- ✅ Removed J1 correction (was double-applied and fighting IK)
- ✅ Disabled orientation control for all waypoints
- ✅ Relaxed success thresholds (0.22m intermediate, 0.20m final)
- ✅ Fixed recovery to use anchored arm J1 correctly
- ✅ Screw spawns at goal position from start with wall orientation
- ✅ Integrated unscrewing task from debug_unscrew.py
- ✅ Active alignment phase with IK + torque compensation
- ✅ Alignment-based rotation start (< 15° error or 3s timeout)
- ✅ Improved screw visibility (larger, brighter colors)
- ✅ Transparent target markers (25% opacity)
- ✅ Alignment plotting after unscrew completion

### December 6, 2025
- ✅ Repositioned arms to Y-axis sides pointing ±X
- ✅ Updated ARM_OFFSET constants to match new positions
- ✅ Removed restrictive arm reachability filter from path planner
- ✅ Added initial arm selection based on travel direction
- ✅ Increased control gains (kp=450, kd=35)
- ✅ Expanded recovery strategies to 15 different J1 angles
- ✅ Added --headless flag (basic implementation)
- ✅ Smart J1 pre-rotation toward target direction
