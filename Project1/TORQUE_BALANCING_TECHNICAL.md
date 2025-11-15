# Torque Balancing Controller - Technical Documentation

## Algorithm Overview

The Torque Balancing Controller extends the standard position controller with an advanced torque-balancing mechanism that minimizes shear forces transmitted to the robot base.

## Theoretical Foundation

### Problem Statement
In a 7-DOF redundant robot arm:
- The base experiences structural shear forces from asymmetric torque distribution
- Wrist joint torques (axes 6 and 7) particularly influence base loading
- Unbalanced torques can cause:
  - Premature bearing wear
  - Vibration and instability
  - Increased stress on mounting interface

### Solution: Null Space Torque Balancing
By leveraging the robot's redundancy, we can apply torque corrections that:
1. Don't affect the primary task (end-effector position/orientation)
2. Live entirely in the null space of the Jacobian
3. Balance the 6th and 7th axis torques to minimize base shear

## Mathematical Formulation

### 1. Primary Task (Position Control)
The primary task follows standard redundant manipulator control:

$$\dot{\mathbf{x}}_d = \mathbf{K}_p \mathbf{e}_p + \mathbf{K}_o \mathbf{e}_o$$

where:
- $\dot{\mathbf{x}}_d$: desired end-effector velocity (6D)
- $\mathbf{K}_p$: position gain matrix
- $\mathbf{K}_o$: orientation gain matrix
- $\mathbf{e}_p$: position error
- $\mathbf{e}_o$: orientation error

### 2. Jacobian Inversion
Map Cartesian velocities to joint velocities using damped least squares:

$$\dot{\mathbf{q}} = \mathbf{J}^+_d \dot{\mathbf{x}}_d + (\mathbf{I} - \mathbf{J}^+_d \mathbf{J}) \mathbf{q}_{null}$$

where:
- $\mathbf{J}^+_d = \mathbf{J}^T(\mathbf{J}\mathbf{J}^T + \lambda\mathbf{I})^{-1}$: damped pseudo-inverse
- $\mathbf{J}$: 6×7 Jacobian matrix
- $\lambda$: damping factor (typically 0.25)
- $\mathbf{I} - \mathbf{J}^+_d \mathbf{J}$: null space projector $\mathbf{N}$

### 3. Torque Calculation
Joint torques from velocity commands (assuming unit-mass system in zero-gravity):

$$\boldsymbol{\tau} = \mathbf{B} \dot{\mathbf{q}}$$

Simplified in our case to:
$$\boldsymbol{\tau}_i = k_i \cdot \text{ctrl}_i$$

where $k_i$ are actuator coefficients.

### 4. Base Shear Force Estimation
The base shear force is estimated as the difference between wrist joint torques:

$$F_{shear} = |\tau_6 - \tau_7|$$

Physical interpretation:
- When $\tau_6 > \tau_7$: net clockwise moment on base
- When $\tau_7 > \tau_6$: net counterclockwise moment
- When $\tau_6 \approx \tau_7$: balanced, minimal shear

### 5. Null Space Correction
Generate a corrective torque vector in the null space:

$$\boldsymbol{\tau}_{corr} = \begin{bmatrix} 0 \\ 0 \\ 0 \\ 0 \\ 0 \\ -k_{balance}(\tau_6 - \tau_7) \\ k_{balance}(\tau_6 - \tau_7) \end{bmatrix}$$

This correction:
- Reduces axis 6 torque
- Increases axis 7 torque (or vice versa)
- Keeps their difference minimal

### 6. Null Space Projection
Project the correction into null space:

$$\boldsymbol{\tau}_{ns} = \mathbf{N} \boldsymbol{\tau}_{corr}$$

where $\mathbf{N} = \mathbf{I} - \mathbf{J}^+ \mathbf{J}$ ensures the correction doesn't affect the Cartesian task.

### 7. Final Control Command
Combine position control with torque balancing:

$$\dot{\mathbf{q}}_{final} = \dot{\mathbf{q}}_{primary} + \alpha \boldsymbol{\tau}_{ns}$$

where $\alpha$ is a scaling factor (typically 0.01) to keep the torque corrections small relative to the primary task.

## Algorithm Pseudocode

```
FUNCTION TorqueBalancingController():
    // Step 1: Compute primary position control (identical to position controller)
    current_pos ← get_end_effector_position()
    target_pos ← get_target_position()
    position_error ← target_pos - current_pos
    
    current_orient ← get_end_effector_orientation()
    target_orient ← get_target_orientation()
    orientation_error ← compute_rotation_error(current_orient, target_orient)
    
    J ← compute_jacobian()  // 6×7 matrix
    
    // Desired Cartesian velocities
    v_desired ← K_pos * position_error + K_orient * orientation_error
    
    // Inverse kinematics with damping
    J_inv ← J^T * (J*J^T + λI)^(-1)
    q_dot_primary ← J_inv * v_desired
    
    // Step 2: Compute torque-based corrections
    tau ← compute_joint_torques()  // From q_dot_primary
    
    // Estimate base shear force
    shear_force ← |tau[5] - tau[6]|  // |tau_6 - tau_7|
    
    // Only apply correction if shear force exceeds threshold
    IF shear_force > THRESHOLD:
        // Create correction vector
        tau_corr ← zeros(7)
        tau_diff ← tau[5] - tau[6]
        tau_corr[5] ← -k_balance * tau_diff
        tau_corr[6] ← k_balance * tau_diff
        
        // Project to null space
        N ← I - J_inv * J
        tau_ns ← N * tau_corr
        
        // Apply correction with scaling
        q_dot_final ← q_dot_primary + α * tau_ns
    ELSE:
        q_dot_final ← q_dot_primary
    END IF
    
    // Step 3: Apply velocity limits and control
    q_dot_final ← clip(q_dot_final, -v_max, v_max)
    data.ctrl ← q_dot_final
    
    RETURN control_data
END FUNCTION
```

## Key Parameters

| Parameter | Default | Range | Effect |
|-----------|---------|-------|--------|
| `kp_position` | 100.0 | 50-200 | Position tracking stiffness |
| `kd_position` | 20.0 | 10-50 | Position damping |
| `max_joint_velocity` | 2.0 | 1-5 rad/s | Velocity saturation limit |
| `torque_balance_gain` | 0.5 | 0.1-1.0 | Strength of torque balancing |
| `shear_force_threshold` | 0.1 | 0.01-1.0 | Activation threshold |
| `lambda_damping` | 0.25 | 0.1-0.5 | Jacobian damping factor |
| `correction_scaling` | 0.01 | 0.001-0.1 | Null space correction scale |

## Performance Analysis

### Metrics Computed

1. **Position Error**: $e_p = \|\mathbf{p}_{target} - \mathbf{p}_{current}\|$

2. **Orientation Error**: $e_o = \frac{1}{2} \text{vex}(\mathbf{R}_{error})$
   - where $\mathbf{R}_{error} = \mathbf{R}_{current}^T \mathbf{R}_{target}$

3. **Base Shear Force**: $F_{shear} = |\tau_6 - \tau_7|$

4. **Energy**: $E = \sum_i \tau_i \dot{q}_i$

5. **Control Effort**: $u = \|\dot{\mathbf{q}}\|_2$

### Stability Considerations

1. **Lyapunov Stability**: The position control component is stable by design (PD controller)
2. **Null Space Independence**: Corrections in null space don't destabilize primary task
3. **Bounded Corrections**: Scaling factor $\alpha$ ensures bounded correction magnitude
4. **Damping**: Jacobian damping prevents singular configuration blow-up

## Comparison with Other Approaches

### vs. Simple Position Control
- **Pros**: Better base loading, reduced structural stress
- **Cons**: Slightly higher computational cost

### vs. Force/Torque Feedback
- **Pros**: No need for force/torque sensors
- **Cons**: Estimated shear force less accurate than measured

### vs. Impedance Control
- **Pros**: Maintains trajectory tracking
- **Cons**: Impedance control allows more compliance

## Practical Recommendations

### When to Use Torque Balancing
✅ High-speed operations (base stress matters more)
✅ Mounted on flexible structures
✅ When bearing wear is a concern
✅ Long-duration operations
✅ Precision applications with mounting sensors

### When to Use Position Control
✅ High-accuracy trajectory following critical
✅ Fast task completion is priority
✅ Robot on stiff, stable base
✅ Torque distribution doesn't matter

### Tuning Guidelines

**To increase balance strength**:
- Increase `torque_balance_gain` (0.1 → 1.0)
- Decrease `shear_force_threshold`

**To maintain tracking while balancing**:
- Increase `kp_position` and `kd_position`
- Decrease `correction_scaling`

**For singularity-prone configurations**:
- Increase `lambda_damping`
- May slightly reduce balance effectiveness

## References

1. Siciliano, B., Sciavicco, L., Villani, L., & Oriolo, G. (2009). Robotics: modelling, planning and control.
2. Nakamura, Y. (1991). Advanced Robotics: Redundancy and Optimization.
3. Khatib, O. (1987). A unified approach for motion and force control of robot manipulators.
4. Chiaverini, S. (1997). Singularity-robust task-priority redundancy resolution using the damped pseudo-inverse.

## Implementation Notes

- The controller assumes zero-gravity environment
- Joint torques are estimated from control commands (not measured)
- Null space projector is computed at each step (O(n³) complexity)
- All data is collected automatically during simulation
- Plots include base shear force evolution if using torque controller
