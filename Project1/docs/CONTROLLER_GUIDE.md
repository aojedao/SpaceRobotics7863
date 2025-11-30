# Controller Module - Quick Reference

## Overview
The `controller.py` module provides a modular, extensible controller framework for the KUKA iiwa14 robot in zero-gravity simulation.

## Available Controllers

### 1. Position Controller
**Class**: `PositionController`
**Alias**: `'position'`

Standard 6DOF Cartesian space control with:
- Position tracking (X, Y, Z)
- Quaternion-based orientation control
- Damped least squares for redundancy handling
- Angular velocity damping

**Usage**:
```bash
python integrated_simulation.py --controller position
```

**Configuration Parameters**:
```python
controller = ControllerFactory.create_controller(
    'position',
    model,
    data,
    kp_position=100.0,        # Position gain
    kd_position=20.0,         # Derivative gain
    max_joint_velocity=2.0    # Max velocity limit
)
```

### 2. Torque Balancing Controller ⭐ NEW
**Class**: `TorqueBalancingController`
**Alias**: `'torque_balancing'`

Advanced controller that minimizes base shear forces through intelligent torque distribution:
- Maintains position/orientation control (primary task)
- Calculates torque difference between 6th and 7th axes
- Uses null space to apply torque corrections without affecting end-effector
- Minimizes shear forces at robot base

**Physical Principle**:
The base shear force is primarily influenced by the difference between the wrist joint torques (axes 6 and 7). By balancing these torques in the null space, we:
- Reduce structural stress on the mounting
- Improve force distribution along the arm
- Maintain trajectory tracking performance

**Usage**:
```bash
python integrated_simulation.py --controller torque_balancing
```

**Configuration Parameters**:
```python
controller = ControllerFactory.create_controller(
    'torque_balancing',
    model,
    data,
    kp_position=100.0,
    kd_position=20.0,
    max_joint_velocity=2.0,
    torque_balance_gain=0.5,      # Strength of torque balancing correction
    shear_force_threshold=0.1     # Threshold for active correction
)
```

**Key Methods**:
- `compute_joint_torques()`: Calculate joint torques from control commands
- `calculate_wrench_at_endeffector()`: Compute force/torque at EE
- `compute_base_shear_force()`: Estimate shear force at base from torque difference
- `compute_null_space_torque_correction()`: Generate null-space corrective torques

## Creating a Custom Controller

### Step 1: Create a subclass
```python
from controller import BaseController

class MyCustomController(BaseController):
    def __init__(self, model, data, **kwargs):
        super().__init__(model, data)
        # Initialize your parameters
        self.custom_param = kwargs.get('custom_param', default_value)
    
    def compute_control(self):
        """Implement your control logic"""
        if not self.enabled:
            return
        
        # Get current state using inherited methods
        current_pos = self.get_end_effector_position()
        target_pos = self.get_target_position()
        J = self.compute_jacobian()
        
        # Your control logic here
        # ...
        
        # Apply control to self.data.ctrl
        self.data.ctrl[:] = your_control_command
        
        # Return data dictionary for logging
        return {
            'current_pos': current_pos,
            'current_orient': current_orient_mat,
            'target_orient': target_orient_mat,
            'position_error': position_error,
            'orientation_error': orientation_error,
            'current_joint_vel': current_joint_vel,
            'current_angular_vel': current_angular_vel,
            # Add any custom data you want to collect
            'custom_metric': custom_value
        }
```

### Step 2: Register in ControllerFactory
```python
class ControllerFactory:
    AVAILABLE_CONTROLLERS = {
        'position': PositionController,
        'torque_balancing': TorqueBalancingController,
        'my_custom': MyCustomController,  # Add your controller
    }
```

### Step 3: Use your controller
```bash
python integrated_simulation.py --controller my_custom
```

## Data Collection

Both controllers return a dictionary with the following structure:

**Standard Data** (all controllers):
```python
{
    'current_pos': np.array([x, y, z]),           # Current end-effector position
    'current_orient': np.array(3x3),              # Current orientation matrix
    'target_orient': np.array(3x3),               # Target orientation matrix
    'position_error': np.array([ex, ey, ez]),     # Position error
    'orientation_error': np.array([ex, ey, ez]),  # Orientation error
    'current_joint_vel': np.array(7,),            # Joint velocities
    'current_angular_vel': np.array([ωx, ωy, ωz])# Angular velocity
}
```

**Additional Data** (TorqueBalancingController):
```python
{
    ...above...
    'shear_force': float,                         # Estimated base shear force
    'torque_diff': float,                         # Torque difference (τ6 - τ7)
    'joint_torques': np.array(7,)                 # Joint torques
}
```

This data is automatically collected by `collect_controller_data()` in the simulation.

## Performance Comparison

| Metric | Position Controller | Torque Balancing Controller |
|--------|-------------------|--------------------------|
| Position Accuracy | ✅ Excellent | ✅ Excellent |
| Orientation Accuracy | ✅ Good | ✅ Good |
| Base Shear Force | ⚠️ Standard | ✅ Minimized |
| Computational Load | ✅ Low | ⚠️ Slightly Higher |
| Structural Stress | ⚠️ Standard | ✅ Reduced |

## Troubleshooting

### Controller not responding
- Check that controller is enabled: `controller.enabled = True`
- Verify model has required joints and sites
- Check for error messages during initialization

### Poor tracking performance
- Adjust controller gains (kp_position, kd_position)
- Increase max_joint_velocity limit
- Check for singularities (look for unrealistic joint velocities)

### Excessive joint velocities with torque controller
- Reduce torque_balance_gain
- Increase damping_factor in Jacobian inversion
- Check shear_force_threshold setting

## API Reference

### BaseController
```python
def get_end_effector_position() -> np.ndarray
def get_end_effector_orientation() -> np.ndarray
def get_target_position() -> np.ndarray
def get_target_orientation() -> np.ndarray
def compute_jacobian() -> np.ndarray
```

### PositionController
```python
def compute_control() -> dict
```

### TorqueBalancingController
```python
def compute_control() -> dict
def compute_joint_torques() -> np.ndarray
def calculate_wrench_at_endeffector() -> np.ndarray
def compute_base_shear_force(joint_torques) -> tuple
def compute_null_space_torque_correction(joint_torques) -> np.ndarray
```

### ControllerFactory
```python
@classmethod
def create_controller(controller_type, model, data, **kwargs) -> BaseController

@classmethod
def get_available_controllers() -> list
```

## Examples

### Example 1: Run position controller for 10 seconds
```bash
python integrated_simulation.py --controller position --duration 10
```

### Example 2: Run torque balancing controller indefinitely
```bash
python integrated_simulation.py --controller torque_balancing
```

### Example 3: Custom controller configuration
```python
from controller import ControllerFactory
import mujoco

# Load model and create data
model = mujoco.MjModel.from_xml_path("model.xml")
data = mujoco.MjData(model)

# Create torque balancing controller with custom gains
controller = ControllerFactory.create_controller(
    'torque_balancing',
    model,
    data,
    kp_position=120.0,
    kd_position=25.0,
    max_joint_velocity=3.0,
    torque_balance_gain=0.75
)

# Simulate
for step in range(10000):
    result = controller.compute_control()
    mujoco.mj_step(model, data)
```

## Notes

- Controllers are instantiated fresh for each simulation run
- Controller state (errors, velocities) is maintained across steps
- Data collection happens automatically in the simulation
- All controllers use the same joint and site naming convention
