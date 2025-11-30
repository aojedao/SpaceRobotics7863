# Manual Joint 7 Control - Usage Guide

## Overview
The simulation now includes a manual actuator (`manual_joint7`) that you can control directly through the MuJoCo viewer interface while the simulation is running.

## How It Works

### Actuator Setup
- **Actuator ID**: 8 (manual_joint7)
- **Joint**: joint7 (robot's 7th joint)
- **Control Range**: -200 to 200 (torque in N·m)
- **Type**: Motor actuator (exposed as a slider in the viewer)

### Controller Protection
- The controller detects the manual actuator automatically
- When applying control commands, the controller **skips writing** to actuator 8
- This allows you to independently control joint 7 via the viewer slider

## Usage Instructions

### Running the Simulation with Manual Control

```bash
# Run with position controller
python integrated_simulation.py --controller position --duration 10

# Run with torque-balancing controller  
python integrated_simulation.py --controller torque_balancing --duration 10

# Run indefinitely (press ESC in viewer to close)
python integrated_simulation.py --controller torque_balancing
```

### Using the Manual Slider in the Viewer

1. **Launch the simulation** (one of the commands above)
2. **Look for the control panel** in the MuJoCo viewer on the left side
3. **Find the slider for `manual_joint7`** (should be near the bottom of the actuator list)
4. **Drag the slider** left/right to apply torque to joint 7:
   - Left (negative): applies negative torque
   - Right (positive): applies positive torque
   - The range is -200 to +200 N·m

### What Happens When You Adjust the Slider

- The manual torque is **immediately applied** to joint 7
- The robot's controller (position or torque-balancing) continues to operate on joints 1-6 and the gripper
- **No conflict**: The controller never overwrites your manual input
- The manual torque is applied **in addition to** any passive effects

## Debug Output

When you start the simulation, you should see:
```
✓ Manual actuator 'manual_joint7' detected at actuator ID 8
```

This confirms the manual control system is active.

## Implementation Details

### In `controller.py`:

1. **BaseController.__init__()**: Detects the manual actuator by name
   ```python
   self.manual_actuator_id = mujoco.mj_name2id(
       self.model, 
       mujoco.mjtObj.mjOBJ_ACTUATOR, 
       "manual_joint7"
   )
   ```

2. **BaseController.apply_control_vector()**: Skips the manual actuator when writing ctrl values
   ```python
   if manual_id is not None and i == manual_id:
       continue  # Don't overwrite manual actuator
   ```

### In `integrated_simulation.py`:

The manual actuator is added to the generated XML:
```xml
<motor name="manual_joint7" joint="joint7" gear="1" ctrlrange="-200 200"/>
```

## Plotting

- Plots are generated **AFTER the simulation ends and the viewer is closed**
- This prevents matplotlib windows from interfering with the MuJoCo viewer
- The plot image is saved to: `integrated_controller_performance_analysis.png`

## Troubleshooting

**Q: I don't see the manual_joint7 slider in the viewer**
- A: The slider is in the "Actuators" section of the control panel. Scroll down if needed.

**Q: Joint 7 is still being controlled by the robot's controller**
- A: Make sure the debug output shows "Manual actuator detected...". If not, the actuator detection failed.

**Q: My manual torque seems to have no effect**
- A: Verify the joint is not locked and that you're moving the slider to non-zero values. Try values like -100 or +100 first.

**Q: How do I know if the manual control is working?**
- A: Watch joint 7 in the simulation while you move the slider. It should move/rotate in response to your slider input.

## Next Steps

Try these experiments:
1. Run the torque-balancing controller and manually add torque to joint 7 to observe how it affects the base moment
2. Adjust the torque-balancing gain in the controller to see how it responds to manual disturbances
3. Collect long-term plots to analyze the effect of manual joint 7 control on the overall system behavior
