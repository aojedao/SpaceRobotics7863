# Wall Crawler MuJoCo Simulation - Copilot Context# Wall Crawler MuJoCo Simulation - Copilot Context



## Project Overview## Project Overview

Dual-arm wall-crawler robot simulation for ISS module traversal using MuJoCo physics engine.Dual-arm wall-crawler robot simulation for ISS module traversal using MuJoCo physics engine.

The robot alternates between two KUKA iiwa14 arms with Robotiq 2F85 grippers to traverse walls.The robot alternates between two KUKA iiwa14 arms with Robotiq 2F85 grippers to traverse walls.



## Current State (December 4, 2025)## Current State (December 3, 2025)



### Test Results### ⚠️ CRITICAL ISSUE - FORCED ACCEPTANCE IS FAKE COMPLETION

**Latest Run: 30% Success Rate (3/10 complete trajectories)**

- Run 2: 4/4 WP complete, 0 recoveries**User Feedback:** "What is forced acceptance? It seems to fake completing a trajectory but it doesn't."

- Run 5: 7/7 WP complete, 1 recovery  

- Run 6: 9/9 WP complete, 2 recoveries**Problem Explanation:**

- 7 runs timed out at later waypoints (WP5-8)The current "forced acceptance" mechanism is a WORKAROUND, not a real solution:

- 25 recovery triggers total across all runs- When the robot gets stuck and can't reach a waypoint within the threshold (e.g., 15-20cm)

- After a timeout, it "accepts" the position even if error is up to 50-70cm away

### ✅ FORCED ACCEPTANCE REMOVED- This marks the waypoint as "complete" but the robot ISN'T actually at the waypoint

- **No more fake completions** - waypoints must be reached within 0.15m threshold- The trajectory "completes" on paper but the robot hasn't actually traversed the path correctly

- Robot either reaches waypoint correctly OR triggers recovery mechanism

- All force acceptance code has been removed (was at 0.50-0.70m thresholds)**Why This Is Bad:**

1. Robot may be 50cm+ away from where it should be anchored

### Recovery Mechanism (Working)2. Subsequent waypoints become harder to reach (error compounds)

When robot gets stuck (20s timeout, error > 0.12m):3. The screw task at the end happens at wrong location

1. Triggers recovery mode (max 5 attempts, 3.0s duration each)4. Success metrics are misleading - "60% success" doesn't mean 60% correct trajectories

2. **Odd attempts**: Use `compute_first_joint_target()` for optimal J1 rotation

3. **Even attempts**: Rotate J1 by 180° ### Current Forced Acceptance Thresholds (THESE ARE TOO LENIENT)

4. Skip anchor force during recovery to allow body repositioning```python

5. Skip moving arm control during recoverysuccess_threshold = 0.15 if is_final else 0.20  # Normal acceptance (15-20cm)

6. If recovery succeeds (error < 0.12m), resume normal operationforced_accept_threshold = 0.50  # Accepts at 50cm if stuck

# After 2+ recovery attempts: accepts at 60cm

### Controller Configuration# After 6000+ steps: accepts at 70cm (!)

```

#### Per-Joint Diagonal Gain Matrix (Higher for base joints)

```python### 180° Recovery Strategy (Implemented but Not Enough)

kp_joint_gains = [1400, 1200, 1000, 800, 600, 500, 400]  # J1 to J7- When stuck, rotates first joint by 180° to escape local minima

```- Sometimes helps, but often robot still can't reach the target

- The IK solver itself may have fundamental issues

#### Joint Limit Avoidance (Weak, prevents singularities)

```python### What Actually Needs To Be Fixed

kp_limit = 50.0  # Weak push away from limits

margin = 0.2  # radians (~11°) from joint limits1. **The IK Controller Itself** - May be stuck in singularities or local minima

```   - Consider using a different IK approach (e.g., CCD, FABRIK)

   - Add null-space optimization to escape singularities

#### Arm Avoidance (Prevents collisions between arms)   - Implement proper singularity detection and avoidance

```python

arm_avoidance_threshold = 0.40  # meters2. **Path Planning** - Some waypoints may be unreachable

arm_avoidance_strength = 1200.0   - The A* planner uses effective_reach = 0.75m but actual IK fails

critical_arm_distance = 0.20  # meters   - Need to validate reachability BEFORE accepting a path

```   - Consider workspace analysis to prune unreachable positions



### Success Thresholds (STRICT - No Forced Acceptance)3. **Remove Forced Acceptance** - It masks the real problem

```python   - Either reach the waypoint properly OR fail honestly

success_threshold = 0.08  # Final waypoint   - Don't fake success with 50cm+ errors

intermediate_threshold = 0.15  # Intermediate waypoints (USER APPROVED)

```### What's Implemented (Partially Working)



### Key Files1. **Random Goal Generation**: Goals randomly selected from 5 walls

2. **Complete Route Visualization**: Color-coded waypoints

#### wall_crawler_mujoco.py (~3107 lines)3. **Trajectory Error Plotting**: Non-blocking with 3-second timeout

Main simulation file with dual KUKA iiwa14 arms.4. **180° Recovery**: Rotates joint1 when stuck (helps sometimes)

5. **Per-Joint Gain Scaling**: [2.0, 1.8, 1.2, 1.0, 1.0, 0.8, 0.6]

Key code locations:

- **Lines 509-527**: Controller gains with diagonal matrix### Technical Configuration

- **Lines 1088-1100**: Joint limit avoidance implementation

- **Lines 2136-2150**: Anchor force skipped during recovery#### IK Controller Gains

- **Lines 2166-2190**: Moving arm control skipped during recovery```python

- **Lines 2202-2248**: Recovery mechanism with J1 rotationkp_position = 400.0          # Position gain

- **Lines 2265-2300**: Stuck detection and recovery triggeringkd_position = 18.0           # Damping gain  

lambda_dls = 0.012           # Damped least squares regularization

#### dual_arm_robot.xmljoint_gain_scale = [2.0, 1.8, 1.2, 1.0, 1.0, 0.8, 0.6]

MuJoCo model with two KUKA iiwa14 arms + Robotiq 2F85 grippers.```



#### run_history.json#### Current (Broken) Success Thresholds

Tracks all test runs (40+ runs, ~24 historical successes).```python

success_threshold = 0.15-0.20  # What it SHOULD be

### Technical Specificationsforced_accept_threshold = 0.50-0.70  # What it ACTUALLY accepts (BAD)

```

#### Robot Configuration

- **Arms**: 2x KUKA iiwa14 (7-DOF each)### Key Files

- **Grippers**: 2x Robotiq 2F85

- **Joint limits**: #### wall_crawler_mujoco.py (~3122 lines)

  - J1, J3, J5: ±170° (±2.967 rad)- Lines 2315-2375: Forced acceptance logic (NEEDS REMOVAL/FIX)

  - J2, J4, J6: ±120° (±2.094 rad)  - `apply_arm_control()`: Per-joint gain scaling

  - J7: ±175° (±3.054 rad)- `_render_visualization_geoms()`: Route visualization



#### MuJoCo Settings#### dual_arm_robot.xml

- **Timestep**: 0.002 seconds- Two KUKA iiwa14 arms + Robotiq 2F85 grippers

- **Gravity**: Zero (space simulation)- Screw body (hidden until spawned)

- **Integrator**: implicitfast

### Running the Simulation

### Running the Simulation```bash

```bashcd /home/aojedao/Documents/NYU/SpaceRobotics/SpaceRobotics7863/Project1

cd /home/user/Documents/NYU/SpaceRobotics/SpaceRobotics7863/Project1conda activate space-robotics 

conda activate space-robotics python wall_crawler_mujoco.py

python wall_crawler_mujoco.py```

```

### TODO - Priority Fixes Needed

### Known Issues / TODO

1. **REMOVE forced acceptance** - Stop faking success

1. **Timeout at Later Waypoints** - 70% of runs time out at WP5-82. **Fix the IK solver** - Address why it gets stuck

   - Recovery works but robot may still struggle to reach target3. **Validate path reachability** - Don't plan unreachable paths

   - Consider: Better path planning, workspace validation4. **Honest failure reporting** - If it can't reach, say so



2. **Recovery Strategy** - Alternating optimal J1 / 180° helps but not always sufficient### ISS Module Bounds

   - Consider: More diverse recovery strategies```python

ISS_MODULE = {

3. **IK Solver Local Minima** - Robot sometimes gets stuck in poor configurations    'x_min': -2.1, 'x_max': 3.9,  # 6m length

   - Consider: Null-space optimization, singularity detection    'y_min': -0.5, 'y_max': 1.7,  # 2.2m width  

    'z_min': 0.1,  'z_max': 2.2   # 2.1m height

### Repository Organization}

```

#### Active Files (Project1/)
- `wall_crawler_mujoco.py` - Main simulation
- `dual_arm_robot.xml` - MuJoCo model
- `run_history.json` - Test run tracking
- `COPILOT_CONTEXT.md` - This file

#### Archived Files (Project1/archive/)
- `brachiation/` - Old brachiation experiments
- `old_simulators/` - Previous simulation versions
- `old_scripts/` - Utility scripts no longer used
- `old_configs/` - Old MuJoCo configurations
- `old_tests/` - Deprecated test files

### Performance Tracking

| Metric | Value |
|--------|-------|
| Latest Success Rate | 30% (3/10) |
| Historical Success | 60% (24/40+) |
| Recovery Triggers | 25 in 10 runs |
| Avg Waypoints | 6-7 per run |
| Main Failure | Timeout at WP5-8 |
