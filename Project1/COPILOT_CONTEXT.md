# Wall Crawler MuJoCo Simulation - Copilot Context

## Project Overview
Dual-arm wall-crawler robot simulation for ISS module traversal using MuJoCo physics engine.

## Current State (December 1, 2025)

### ✅ What's Working
1. **Body Positioning**: Robot body correctly positioned inside ISS module at (1.00, 1.10, 1.25)
2. **Path Planning**: A* path planning works - finds 6-step path from front wall to back wall
3. **Right Arm IK**: Successfully reaches target with ~0.05m error
   - Target: (1.00, 1.65, 1.40) on front wall
   - Achieved: (1.03, 1.65, 1.36) - within threshold!
4. **Body Locking**: Body stays fixed during arm movements
5. **State Machine**: Transitions correctly through settling → moving_right → goal_reached

### ❌ Current Problem: Left Arm IK
The left arm is NOT converging to its target. It gets stuck in a local minimum.

**Left Arm Target**: (1.00, 1.65, 1.10) - front wall, Z=1.10
**Left Arm Ends Up**: ~(1.11, 1.02, 1.64) or similar - going BACKWARD and UP

The IK is finding solutions where:
- Y goes backward (toward 1.0) instead of forward (toward 1.65)
- Z goes up (toward 1.6-2.0) instead of down (toward 1.10)

### Key Configuration That Works for Right Arm
```python
# RIGHT ARM - This configuration WORKS
self.data.qpos[22] = 0.0     # joint1 - no rotation
self.data.qpos[23] = -1.2    # joint2 - pitch forward (negative for right arm due to 180° rotation)
self.data.qpos[24] = 0.0     # joint3 
self.data.qpos[25] = 0.8     # joint4 - elbow bent UP (positive)
self.data.qpos[26] = 0.0     # joint5 
self.data.qpos[27] = -0.5    # joint6 - wrist angled
self.data.qpos[28] = 0.0     # joint7
```

### What's Been Tried for Left Arm
Multiple joint configurations tested:
- Various joint2 values: 1.0, 1.2, 1.3, 1.35, 1.4
- Various joint4 values: -0.3, -0.5, -0.7, -0.8, -1.0, 0.4
- Various joint6 values: -0.5, -0.2, 0.0, 0.3, 0.4, 0.5

The left arm consistently goes BACKWARD instead of FORWARD.

### Technical Details

#### ISS Module Bounds
```python
ISS_MODULE = {
    'x_min': -2.1, 'x_max': 3.9,
    'y_min': -0.5, 'y_max': 1.7,
    'z_min': 0.1,  'z_max': 2.2,
}
```

#### Robot Configuration
- **KUKA iiwa14** dual arms with Robotiq 2F85 grippers
- **Left arm offset**: -0.25m in X from body center
- **Right arm offset**: +0.25m in X from body center
- **Right arm rotation**: 180° around Z (quat="0 0 0 1" in XML)
- **KUKA reach**: 1.25m (including gripper)

#### IK Controller Settings
```python
self.kp_position = 150.0      # Position control gain
self.kd_position = 40.0       # Damping
self.lambda_dls = 0.08        # Damped least squares regularization
```

#### Key Files
- `wall_crawler_mujoco.py`: Main simulation (lines 1170-1200 for arm config)
- `dual_arm_robot.xml`: MuJoCo model

### Path Waypoints (Current Test)
1. (1.00, 1.65, 1.10) - LEFT arm anchor (front wall)
2. (1.00, 1.65, 1.40) - RIGHT arm target (front wall)
3. (1.30, 0.80, 2.10) - LEFT arm (ceiling)
4. (1.60, 0.20, 2.10) - RIGHT arm (ceiling)
5. (1.90, -0.50, 1.40) - LEFT arm (back wall)
6. (1.90, -0.50, 1.10) - RIGHT arm (back wall)

### Next Steps to Try
1. **Different left arm initial pose**: The arm starts pointing wrong direction
2. **Use nullspace optimization**: Add secondary objective to bias joint motion
3. **Try joint1 rotation**: Rotate the arm base to face forward
4. **Check arm mounting**: Left arm might need different joint2 sign convention
5. **Consider pre-computing IK**: Use analytical IK to find a good starting configuration

### Hypothesis
The left arm's kinematic chain may need joint1 to rotate the base to point toward +Y direction, since the arm naturally points in the +X direction at home position.

### Running the Simulation
```bash
cd /home/aojedao/Documents/NYU/SpaceRobotics/SpaceRobotics7863/Project1
conda activate space-robotics
timeout 60 python wall_crawler_mujoco.py 2>&1
```

### Key Functions in wall_crawler_mujoco.py
- `compute_arm_control()`: Position-only damped least squares IK (lines ~700-770)
- `apply_arm_control()`: Applies IK commands to arm (lines ~770-810)
- `maintain_anchor()`: Keeps body and left arm locked (lines ~810-870)
- `lock_body_position()`: Called after settling to lock positions (lines ~860-870)
- Initial arm configuration: lines ~1175-1200

### Success Criteria
- Left arm error < 0.05m (currently ~0.8-1.0m)
- Right arm error < 0.05m ✅ (achieved 0.050m)
- Both arms reaching their targets before transitioning to next crawl step
