# Wall Crawler MuJoCo Simulation - Copilot Context

## Project Overview

Dual-arm wall-crawler robot simulation for ISS module traversal using MuJoCo physics engine.
The robot alternates between two KUKA iiwa14 arms with Robotiq 2F85 grippers to traverse walls.

## Current State (December 6, 2025)

### Branch: `DualArmDev-side-mounted-arms_antigravity`

### Arm Configuration (UPDATED)
- **Left Arm**: Position Y=-0.25, Euler "0 -90 0" → Points in **-X direction**
- **Right Arm**: Position Y=+0.25, Euler "0 90 0" → Points in **+X direction**
- Arms are on the Y-axis sides of the body box, pointing outward in opposite X directions
- Zero gravity environment (anti-gravity mode)

### Test Results
**Current Success Rate: 2.4% (2/84 runs)**
- Most failures occur at early waypoints (WP1-WP3)
- Best recent run: 12/13 waypoints reached
- 2 complete successes recorded

### Key Parameters

#### Arm Offsets (in wall_crawler_mujoco.py)
```python
LEFT_ARM_OFFSET = -0.25   # Y-axis offset for left arm
RIGHT_ARM_OFFSET = 0.25   # Y-axis offset for right arm
```

#### Control Gains
```python
kp_position = 450.0      # Position control gain (INCREASED)
kd_position = 35.0       # Velocity damping (INCREASED)
kp_orientation = 50.0    # Orientation control
kd_orientation = 20.0
lambda_dls = 0.008       # Damped least squares regularization
```

#### Path Planner
```python
step_size = 0.5m         # Distance between waypoints
effective_reach = 0.75m  # Planning reach (< actual 1.25m arm reach)
```

### Recovery Mechanism
- **15 recovery strategies** with varied J1 angles: [0.0, -1.0, 1.0, -2.0, 2.0, -1.5, 1.5, -2.5, 2.5, 0.5, -0.5, -3.0, 3.0, π/2, -π/2]
- **Stuck threshold**: 8 seconds
- **Recovery duration**: 3 seconds per attempt
- **Smart J1 pre-rotation**: Initial joint 1 angle set toward target direction

### Path Planning Logic
1. **Initial arm selection**: Based on direction to goal
   - Moving -X → LEFT arm starts (points -X)
   - Moving +X → RIGHT arm starts (points +X)
2. **Arm alternation**: Simple alternating LEFT/RIGHT for bipedal locomotion
3. **No arm reachability filter**: Removed to allow direct paths; IK handles reachability at execution

### ISS Module Bounds
```python
ISS_MODULE = {
    'x_min': -2.1, 'x_max': 3.9,
    'y_min': -0.5, 'y_max': 1.7,
    'z_min': 0.1,  'z_max': 2.2
}
```

## Files Overview

| File | Purpose |
|------|---------|
| `wall_crawler_mujoco.py` | Main simulation (~3400 lines) with path planning, IK control, locomotion |
| `dual_arm_robot.xml` | MuJoCo model with robot and ISS environment |
| `run_multiple_tests.py` | Batch test runner (parallel or sequential) |
| `run_history.json` | Test results history |

## Command Line Usage

```bash
# Run with visualization (default)
conda run -n space-robotics python wall_crawler_mujoco.py

# Run headless (basic - needs full locomotion loop)
conda run -n space-robotics python wall_crawler_mujoco.py --headless

# Run without recording to history
conda run -n space-robotics python wall_crawler_mujoco.py --no-record

# Run multiple tests with visualization (sequential)
conda run -n space-robotics python run_multiple_tests.py 5 1 --visualize
```

## Known Issues

1. **Low Success Rate (2.4%)**: IK solver gets stuck in local minima
2. **Arm Crossing**: When arms cross paths, recovery is difficult
3. **Headless Mode Incomplete**: `run_headless()` doesn't run full locomotion loop
4. **Early Waypoint Failures**: Most failures at WP1-WP3 suggest body positioning issues

## TODO

1. [ ] Extract locomotion loop from `run_visualization()` into reusable method
2. [ ] Implement full headless mode with complete locomotion logic
3. [ ] Improve IK convergence with null-space optimization
4. [ ] Better initial body positioning based on first waypoint
5. [ ] Achieve 70% success rate target

## Recent Changes (December 6, 2025)

1. ✅ Repositioned arms to Y-axis sides pointing ±X
2. ✅ Updated ARM_OFFSET constants to match new positions
3. ✅ Removed restrictive arm reachability filter from path planner
4. ✅ Added initial arm selection based on travel direction
5. ✅ Increased control gains (kp=450, kd=35)
6. ✅ Expanded recovery strategies to 15 different J1 angles
7. ✅ Added --headless flag (basic implementation)
8. ✅ Smart J1 pre-rotation toward target direction
