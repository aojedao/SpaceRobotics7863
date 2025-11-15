# Moment Minimization Controller - Technical Documentation

## Overview

The updated `TorqueBalancingController` now uses a **potential field-based moment minimization** approach instead of simple torque difference balancing. This is analogous to obstacle avoidance algorithms but applied to base moment reduction.

## Mathematical Formulation

### Problem Statement

In the manipulator base, the net moment is influenced by:
1. **Direct contribution**: The 7th axis (wrist) torque τ₇
2. **Projected contribution**: How the 6th axis moment propagates through the kinematic chain to the base

By minimizing their difference, we reduce the net moment experienced at the base structure.

### Potential Field Cost Function

The controller minimizes the following cost function:

$$U(\mathbf{q}) = \frac{1}{2}(\tau_7 - M_{proj,6})^2$$

where:
- **τ₇**: Torque command on the 7th axis (joint 7, wrist rotation)
- **M_{proj,6}**: Projected moment at base from 6th axis torque

This is analogous to the obstacle avoidance potential field:
$$U_{obstacle} = \frac{1}{2}(d_{min} - d)^2$$

### Projected Moment Calculation

The moment from the 6th axis projects to the base through the kinematic chain:

$$M_{proj,6} = \|\mathbf{J}_{rot,6}\| \cdot \tau_6$$

where:
- **J_{rot,6}**: Column 6 of the rotational Jacobian (3×1 vector)
- **τ₆**: Torque command on the 6th axis
- **‖·‖**: Euclidean norm

The rotational Jacobian column represents how the 6th joint's angular velocity affects the base moment. By scaling by τ₆, we get the magnitude of moment that propagates to the base.

### Cost Gradient (Obstacle Avoidance Analogy)

**Obstacle Avoidance Gradient**:
$$\nabla U_{obs} = -(d_{min} - d) \cdot \nabla d$$

**Moment Minimization Gradient**:
$$\nabla U_{moment} = (\tau_7 - M_{proj,6}) \cdot \nabla M_{proj,6}$$

The controller uses this gradient in the null space to drive corrections:

$$\Delta \mathbf{q}_{null} = -\alpha \cdot \mathbf{N} \cdot \nabla U_{moment}$$

where:
- **α**: Gain factor (torque_balance_gain)
- **N**: Null space projector = I - J⁺J
- **∇U_{moment}**: Gradient of moment cost

## Algorithm Details

### 1. Moment Error Calculation

```python
moment_error = τ₇ - M_{proj,6}
```

**Interpretation**:
- **moment_error > 0**: 7th axis torque dominates → need to reduce τ₇ or increase M_{proj,6}
- **moment_error < 0**: Projected moment dominates → need to increase τ₇ or decrease M_{proj,6}
- **moment_error ≈ 0**: Moment balanced → minimal base loading

### 2. Null Space Correction Vector

```python
correction_vector[5] = -moment_error × gain × ‖J_{rot,6}‖
correction_vector[6] = -moment_error × gain
```

This creates corrections that:
- Adjust τ₆ proportionally to the magnitude of the Jacobian (affects how much moment projects)
- Adjust τ₇ directly to balance the moment

### 3. Null Space Projection

```python
null_space_correction = N × correction_vector = (I - J⁺J) × correction_vector
```

Ensures corrections don't interfere with primary end-effector task.

### 4. Control Law

```python
v_final = v_primary + α_correction × null_space_correction
```

Primary position/orientation control is maintained, with small null-space corrections added.

## Implementation Methods

### `compute_projected_moment_to_base()`

Calculates how the 6th axis torque manifests as moment at the base.

**Returns**:
- `projected_moment`: Magnitude of moment from 6th axis at base
- `j6_rot`: Rotational Jacobian column for 6th joint
- `tau_6`: Current 6th axis torque

**Key Formula**:
```
projected_moment = ‖J_rot[:,5]‖ × |τ₆|
```

### `compute_base_moment_cost()`

Evaluates the potential field cost and its gradient.

**Returns Dictionary**:
- `cost`: Potential field value (½ × moment_error²)
- `moment_error`: τ₇ - M_{proj,6}
- `tau_7`: Current 7th axis torque
- `tau_6`: Current 6th axis torque
- `projected_moment`: M_{proj,6}
- `cost_grad_tau7`: ∂U/∂τ₇ = moment_error
- `cost_grad_tau6`: ∂U/∂τ₆ (how τ₆ affects cost)
- `j6_rot`: J_rot[:,5]

### `compute_null_space_torque_correction()`

Generates corrective velocity commands in the null space to minimize moment cost.

**Strategy**:
1. If moment_error > 0 (τ₇ dominates):
   - Reduce τ₆ to decrease projected moment
   - Reduce τ₇ directly
2. If moment_error < 0 (M_{proj,6} dominates):
   - Increase τ₆ to increase projected moment
   - Increase τ₇ directly
3. Apply all corrections in null space to not affect primary task

## Data Collection and Monitoring

The controller returns rich data for analysis:

```python
{
    'current_pos': np.array([x, y, z]),
    'current_orient': 3×3 rotation matrix,
    'target_orient': 3×3 rotation matrix,
    'position_error': position error vector,
    'orientation_error': orientation error vector,
    'current_joint_vel': joint velocities,
    'current_angular_vel': end-effector angular velocity,
    'moment_cost': U = ½(τ₇ - M_{proj,6})²,
    'moment_error': τ₇ - M_{proj,6},
    'tau_6': 6th axis torque,
    'tau_7': 7th axis torque,
    'projected_moment_to_base': M_{proj,6},
    'joint_torques': all 7 joint torques
}
```

### Key Metrics to Monitor

1. **moment_error**: Should converge to ~0
2. **moment_cost**: Should decrease over time
3. **tau_7 vs projected_moment_to_base**: Should approach equality
4. **position_error**: Must remain acceptable (primary task)

## Comparison: Old vs New Approach

### Old Approach
```
Cost: |τ₆ - τ₇|  (simple difference)
```
- ✅ Simple to implement
- ❌ Doesn't account for how torques actually propagate
- ❌ Equal weighting of both axes regardless of kinematic chain

### New Approach
```
Cost: ½(τ₇ - M_{proj,6})²  where M_{proj,6} = ‖J_rot[:,5]‖ × τ₆
```
- ✅ Physics-based: accounts for kinematic transmission
- ✅ Potential field analogy: familiar from obstacle avoidance
- ✅ Gradient-driven: smooth convergence
- ✅ Smooth cost function: differentiable for optimization
- ✅ Prioritizes what matters: base moment reduction

## Tuning Parameters

| Parameter | Default | Role |
|-----------|---------|------|
| `torque_balance_gain` | 0.5 | Strength of null-space correction (0.0-1.0) |
| `shear_force_threshold` | 0.1 | Activation threshold for corrections |
| `correction_scaling` | 0.01 | How much correction affects final velocities |

### Tuning Guidelines

**To increase moment balancing aggressiveness**:
- ↑ `torque_balance_gain` (0.5 → 0.8)
- ↓ `correction_scaling` threshold
- Monitor that tracking error doesn't increase

**To maintain trajectory tracking priority**:
- ↓ `correction_scaling` (0.01 → 0.005)
- ↑ `torque_balance_gain` (compensate by being more selective)

**For different manipulator sizes**:
- Scale gains based on arm length (L)
- Moment projections scale with L: M ∝ L × F

## Physical Interpretation

### Why This Works

1. **Kinematic Transmission**: The rotational Jacobian column ‖J_{rot,6}‖ captures how the 6th joint's motion affects base moment through the entire kinematic chain.

2. **Moment Balancing**: By making τ₇ ≈ M_{proj,6}, we ensure:
   - The 7th axis (wrist) torque balances the propagated moment
   - Net moment at base is minimized
   - Structure experiences less dynamic loading

3. **Null Space Safety**: All corrections live in null space:
   - Primary task (position/orientation) unaffected
   - Redundancy utilized for secondary objective
   - Graceful degradation if base moment can't be fully balanced

### Practical Implications

- **Bearing Wear**: Reduced base moment → longer bearing life
- **Vibration**: More balanced loading → less vibration
- **Mounting Stress**: Distributed loading → safer structure
- **Energy**: More efficient force distribution

## Example Usage

```python
from controller import ControllerFactory

# Create controller
controller = ControllerFactory.create_controller(
    'torque_balancing',
    model,
    data,
    kp_position=100.0,
    kd_position=20.0,
    torque_balance_gain=0.5
)

# Simulation loop
for step in range(10000):
    result = controller.compute_control()
    
    # Monitor moment balance
    moment_error = result['moment_error']
    cost = result['moment_cost']
    
    if step % 100 == 0:
        print(f"Moment error: {moment_error:.6f}, Cost: {cost:.6f}")
    
    mujoco.mj_step(model, data)
```

## Visualization and Analysis

The collected data allows for insightful plots:

1. **Moment Cost vs Time**: Shows convergence of cost function
2. **τ₇ vs M_{proj,6}**: Shows how well moments are balanced
3. **moment_error**: Should approach zero
4. **Position Tracking**: Verify primary task performance
5. **Joint Torques**: Understand torque distribution

Example comparison plot:
```
subplot(2,2,1): moment_cost (should decrease)
subplot(2,2,2): moment_error (should → 0)
subplot(2,2,3): tau_7 vs projected_moment_to_base (should converge)
subplot(2,2,4): position_error (should stay acceptable)
```

## References and Related Work

This approach combines:
1. **Artificial Potential Fields** (Khatib, 1986)
2. **Redundancy Resolution** (Siciliano et al., 2009)
3. **Null Space Projection** (Chiaverini, 1997)
4. **Kinematic Jacobian Analysis** (Craig, 2005)

The moment projection concept is derived from:
- Humanoid robotics base moment minimization
- Bipedal walking research (limit cycles with moment minimization)
- Space robot base reaction minimization (zero-g operations)
