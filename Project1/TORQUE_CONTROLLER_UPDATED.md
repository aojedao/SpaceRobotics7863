# Updated Torque Balancing Controller - Technical Reference

## Quick Summary of Changes

The controller has been updated from a simple torque-difference approach to a **physics-based potential field method** that minimizes the moment perceived at the robot base.

### Old Method
- Simple torque difference: minimize |τ₆ - τ₇|
- Treats both axes equally regardless of kinematics

### New Method  
- Potential field cost: minimize ½(τ₇ - M_{proj,6})²
- Accounts for kinematic moment transmission
- Uses gradient-based null space correction

## Core Mathematical Concepts

### 1. Base Moment from Wrist Joints

The moment at the robot base is affected by:

**From 7th axis (direct)**:
$$M_{direct} = \tau_7$$

**From 6th axis (through kinematic chain)**:
$$M_{projected} = \|\mathbf{J}_{rot,6}\| \cdot \tau_6$$

where J_{rot,6} is column 6 of the rotational Jacobian, representing how 6th joint motion affects base angular velocity.

### 2. Potential Field Cost Function

$$U(\mathbf{q}) = \frac{1}{2}(\tau_7 - M_{proj,6})^2$$

**Properties**:
- Smooth and differentiable
- Minimum at τ₇ = M_{proj,6} (perfect balance)
- Quadratic penalty for imbalance
- Zero cost when moments are balanced

**Analogy to Obstacle Avoidance**:

| Obstacle Avoidance | Moment Minimization |
|-------------------|-------------------|
| $U = \frac{1}{2}(d_{min} - d)^2$ | $U = \frac{1}{2}(\tau_7 - M_{proj})^2$ |
| Minimize distance to obstacle | Minimize moment difference |
| $\nabla d$ guides away from obstacle | $\nabla U$ guides to balance moment |
| Null space used for avoidance | Null space used for balancing |

### 3. Cost Gradient and Null Space Correction

**Cost gradient**:
$$\nabla U = (\tau_7 - M_{proj,6}) \cdot \begin{bmatrix} -\|\mathbf{J}_{rot,6}\| \\ 1 \end{bmatrix}$$

where the first component affects τ₆ and second affects τ₇.

**Null space correction**:
$$\Delta \mathbf{q}_{null} = -\alpha \cdot \mathbf{N} \cdot \nabla U$$

where:
- α: gain factor
- N = I - J⁺J: null space projector

The negative sign ensures we move to minimize cost (gradient descent).

## Implementation Flow

```
STEP 1: Compute Primary Task (Position Control)
  ↓
STEP 2: Compute Joint Torques from Control Commands
  ↓
STEP 3: Project 6th Axis Moment to Base
  M_proj = ||J_rot[:,5]|| × τ_6
  ↓
STEP 4: Calculate Moment Error
  error = τ_7 - M_proj
  ↓
STEP 5: Evaluate Cost
  U = 0.5 × error²
  ↓
STEP 6: Compute Gradient-Based Correction in Null Space
  correction = -gain × N × gradient(U)
  ↓
STEP 7: Combine Primary Task + Null Space Correction
  v_final = v_primary + scale × correction
  ↓
STEP 8: Apply to Joints and Simulate
```

## Detailed Method Descriptions

### `compute_projected_moment_to_base()`

**Purpose**: Calculate how 6th axis torque manifests as moment at base

**Process**:
1. Extract rotational Jacobian column for joint 6: `j6_rot = J_rot[:, 5]`
2. Get current 6th axis torque: `tau_6 = ctrl[5]`
3. Compute projection: `proj_moment = ||j6_rot|| × |tau_6|`

**Physical Meaning**:
- `||j6_rot||` ≈ "moment transmission ratio" from joint 6 to base
- In singular configurations: ratio can be very high
- In well-conditioned configurations: ratio is moderate

**Returns**:
- `projected_moment`: Scalar magnitude of moment at base from τ₆
- `j6_rot`: 3-element vector (rotational Jacobian column)
- `tau_6`: Current torque on axis 6

### `compute_base_moment_cost()`

**Purpose**: Evaluate potential field cost and all relevant data

**Process**:
1. Get τ₇ from control commands
2. Compute M_{proj} using projected moment function
3. Calculate error: `e = τ₇ - M_{proj}`
4. Compute cost: `U = 0.5 × e²`
5. Compute gradients for optimization

**Cost Gradient Meanings**:
- `cost_grad_tau7`: How much decreasing τ₇ would reduce cost = e
- `cost_grad_tau6`: How much changing τ₆ would affect cost ≈ -e × ||j6_rot||

**Returns Dictionary with**:
- `cost`: The potential field value
- `moment_error`: τ₇ - M_{proj,6}
- `tau_7`, `tau_6`: Current torques
- `projected_moment`: M_{proj,6}
- `cost_grad_tau7`, `cost_grad_tau6`: Gradient components
- `j6_rot`: For further analysis

### `compute_null_space_torque_correction()`

**Purpose**: Generate corrective velocities that reduce cost while preserving primary task

**Algorithm**:

1. **Compute null space projector**:
   ```
   N = I - J⁺ × J
   ```

2. **Get cost gradient information**:
   ```
   moment_error = τ₇ - M_{proj,6}
   j6_rot = rotational Jacobian column for joint 6
   ```

3. **Create correction vector**:
   ```
   If |moment_error| > threshold:
       correction[5] = -moment_error × gain × ||j6_rot||  # Joint 6
       correction[6] = -moment_error × gain               # Joint 7
   Else:
       correction = 0  (no correction needed)
   ```

4. **Project to null space**:
   ```
   null_space_correction = N × correction
   ```

5. **Store debug data**:
   ```
   Save to _last_correction_data for analysis
   ```

**Why This Works**:

- **Correcting Joint 6**: Adjusting τ₆ changes how much moment projects to base
  - If e > 0 (τ₇ too high): reduce τ₆ → reduce M_{proj} → balance
  - If e < 0 (M_{proj} too high): increase τ₆ → increase M_{proj} → balance
  
- **Correcting Joint 7**: Directly adjusts the dominant moment term
  - If e > 0 (τ₇ too high): reduce τ₇
  - If e < 0 (τ₇ too low): increase τ₇

- **Null Space**: These corrections don't affect end-effector position/orientation
  - All corrections are in the null space by construction
  - Primary task fully preserved

## Control Update Equation

```python
v_final = v_primary + correction_scaling × null_space_correction

where:
  v_primary = position/orientation control (unchanged)
  null_space_correction = (I - J⁺J) × gradient_correction
  correction_scaling = 0.01 (tunable parameter)
```

**Why small scaling factor?**
- Primary task must be prioritized
- Moment minimization is secondary objective
- Too large scaling → poor tracking, minimal moment reduction
- Too small scaling → good tracking, minimal moment reduction (can still converge)

## Data Flow and Output

```python
controller.compute_control() returns:
{
    # Position/Orientation Tracking
    'position_error': position deviation,
    'orientation_error': orientation deviation,
    
    # Moment Minimization (NEW)
    'moment_cost': U = 0.5 × (τ₇ - M_{proj})²,
    'moment_error': τ₇ - M_{proj},
    'tau_6': current 6th axis torque,
    'tau_7': current 7th axis torque,
    'projected_moment_to_base': M_{proj},
    
    # Joint States
    'joint_torques': all 7 torques,
    'current_joint_vel': joint velocities,
    'current_angular_vel': end-effector angular velocity
}
```

## Expected Behavior During Simulation

### Early Phase (first 1-2 seconds)
- **moment_cost**: Relatively high, noisy
- **moment_error**: Large positive or negative
- **position_error**: Decreasing (primary task settling)
- **tau_7 vs projected_moment**: Mismatched, diverging

### Middle Phase (2-5 seconds)
- **moment_cost**: Trending downward
- **moment_error**: Oscillating but decreasing in amplitude
- **position_error**: Small, stabilized
- **tau_7 vs projected_moment**: Starting to correlate

### Late Phase (>5 seconds)
- **moment_cost**: Small, relatively stable
- **moment_error**: ~0 ± small oscillations
- **position_error**: Minimal, steady-state
- **tau_7 vs projected_moment**: Well balanced, moving together

## Singularity Considerations

When the manipulator approaches a singular configuration:
- Jacobian determinant → 0
- Null space → large
- ||J_rot|| → very large
- **projected_moment** becomes very sensitive to τ₆

**Handling**:
1. Damping in Jacobian inversion (`lambda_damping = 0.25`)
2. Bounded corrections via `correction_scaling`
3. Primary task prioritization via `J_damped_pinv`

All these prevent "singularity explosion" of moment corrections.

## Stability Analysis

### Lyapunov Function
The moment cost serves as a Lyapunov function for the null-space dynamics:

$$V = U = \frac{1}{2}(\tau_7 - M_{proj,6})^2$$

**V ≥ 0** everywhere, with V = 0 only when τ₇ = M_{proj,6}

**Stability guaranteed by**:
1. Null space projection doesn't affect primary stability (position control is stable)
2. Quadratic cost is convex (no local minima other than global minimum)
3. Negative feedback via gradient descent

### When Stability Could Fail
- If `correction_scaling` is too large → interferes with primary task stability
- If `torque_balance_gain` > 1.0 → overshoot and oscillation
- If manipulator in singularity and correction unbounded → large velocities

**Prevention**: All parameters tuned conservatively, corrections bounded by velocity limits.

## Comparison with Alternative Approaches

| Approach | Cost Function | Pros | Cons |
|----------|--------------|------|------|
| **Simple Difference** | \|τ₆ - τ₇\| | Simple | Ignores kinematics |
| **Weighted Difference** | a\|τ₆\| + b\|τ₇\| | Tunable weights | Still ignores chain |
| **Our Method** | ½(τ₇ - M_{proj})² | Physics-based, smooth | Slightly more complex |
| **Full Dynamics** | F = Ma with full model | Most accurate | Very complex, slow |

## References

1. **Potential Fields**: Khatib, O. (1986). "Real-Time Obstacle Avoidance for Manipulators and Mobile Robots"
2. **Redundancy Resolution**: Siciliano, B., et al. (2009). "Robotics: Modelling, Planning and Control"  
3. **Null Space Projection**: Chiaverini, S. (1997). "Singularity-robust task-priority redundancy resolution"
4. **Humanoid Moment Control**: Kaneko, K., et al. (2008). "Humanoid Robot HRP-3"
5. **Space Robotics Base Reaction**: Umetani, Y., et al. (1989). "Resolved Motion Rate Control of Space Manipulators"

