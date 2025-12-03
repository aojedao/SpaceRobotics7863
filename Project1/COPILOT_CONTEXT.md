# Wall Crawler MuJoCo Simulation - Copilot Context

## Project Overview
Dual-arm wall-crawler robot simulation for ISS module traversal using MuJoCo physics engine.
The robot alternates between two KUKA iiwa14 arms with Robotiq 2F85 grippers to traverse walls.

## Current State (December 3, 2025)

### ⚠️ CRITICAL ISSUE - FORCED ACCEPTANCE IS FAKE COMPLETION

**User Feedback:** "What is forced acceptance? It seems to fake completing a trajectory but it doesn't."

**Problem Explanation:**
The current "forced acceptance" mechanism is a WORKAROUND, not a real solution:
- When the robot gets stuck and can't reach a waypoint within the threshold (e.g., 15-20cm)
- After a timeout, it "accepts" the position even if error is up to 50-70cm away
- This marks the waypoint as "complete" but the robot ISN'T actually at the waypoint
- The trajectory "completes" on paper but the robot hasn't actually traversed the path correctly

**Why This Is Bad:**
1. Robot may be 50cm+ away from where it should be anchored
2. Subsequent waypoints become harder to reach (error compounds)
3. The screw task at the end happens at wrong location
4. Success metrics are misleading - "60% success" doesn't mean 60% correct trajectories

### Current Forced Acceptance Thresholds (THESE ARE TOO LENIENT)
```python
success_threshold = 0.15 if is_final else 0.20  # Normal acceptance (15-20cm)
forced_accept_threshold = 0.50  # Accepts at 50cm if stuck
# After 2+ recovery attempts: accepts at 60cm
# After 6000+ steps: accepts at 70cm (!)
```

### 180° Recovery Strategy (Implemented but Not Enough)
- When stuck, rotates first joint by 180° to escape local minima
- Sometimes helps, but often robot still can't reach the target
- The IK solver itself may have fundamental issues

### What Actually Needs To Be Fixed

1. **The IK Controller Itself** - May be stuck in singularities or local minima
   - Consider using a different IK approach (e.g., CCD, FABRIK)
   - Add null-space optimization to escape singularities
   - Implement proper singularity detection and avoidance

2. **Path Planning** - Some waypoints may be unreachable
   - The A* planner uses effective_reach = 0.75m but actual IK fails
   - Need to validate reachability BEFORE accepting a path
   - Consider workspace analysis to prune unreachable positions

3. **Remove Forced Acceptance** - It masks the real problem
   - Either reach the waypoint properly OR fail honestly
   - Don't fake success with 50cm+ errors

### What's Implemented (Partially Working)

1. **Random Goal Generation**: Goals randomly selected from 5 walls
2. **Complete Route Visualization**: Color-coded waypoints
3. **Trajectory Error Plotting**: Non-blocking with 3-second timeout
4. **180° Recovery**: Rotates joint1 when stuck (helps sometimes)
5. **Per-Joint Gain Scaling**: [2.0, 1.8, 1.2, 1.0, 1.0, 0.8, 0.6]

### Technical Configuration

#### IK Controller Gains
```python
kp_position = 400.0          # Position gain
kd_position = 18.0           # Damping gain  
lambda_dls = 0.012           # Damped least squares regularization
joint_gain_scale = [2.0, 1.8, 1.2, 1.0, 1.0, 0.8, 0.6]
```

#### Current (Broken) Success Thresholds
```python
success_threshold = 0.15-0.20  # What it SHOULD be
forced_accept_threshold = 0.50-0.70  # What it ACTUALLY accepts (BAD)
```

### Key Files

#### wall_crawler_mujoco.py (~3122 lines)
- Lines 2315-2375: Forced acceptance logic (NEEDS REMOVAL/FIX)
- `apply_arm_control()`: Per-joint gain scaling
- `_render_visualization_geoms()`: Route visualization

#### dual_arm_robot.xml
- Two KUKA iiwa14 arms + Robotiq 2F85 grippers
- Screw body (hidden until spawned)

### Running the Simulation
```bash
cd /home/aojedao/Documents/NYU/SpaceRobotics/SpaceRobotics7863/Project1
conda activate space-robotics 
python wall_crawler_mujoco.py
```

### TODO - Priority Fixes Needed

1. **REMOVE forced acceptance** - Stop faking success
2. **Fix the IK solver** - Address why it gets stuck
3. **Validate path reachability** - Don't plan unreachable paths
4. **Honest failure reporting** - If it can't reach, say so

### ISS Module Bounds
```python
ISS_MODULE = {
    'x_min': -2.1, 'x_max': 3.9,  # 6m length
    'y_min': -0.5, 'y_max': 1.7,  # 2.2m width  
    'z_min': 0.1,  'z_max': 2.2   # 2.1m height
}
```
