# Visual Reference: Moment Minimization Algorithm

## Control Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                   TorqueBalancingController                     │
│                      compute_control()                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
        ┌───────────▼──────────┐   ┌──▼──────────────────┐
        │ PRIMARY TASK         │   │ MOMENT MINIMIZATION │
        │ Position/Orientation │   │ Secondary Objective │
        └──────────┬───────────┘   └──────────┬──────────┘
                   │                          │
        ┌──────────▼───────────┐   ┌──────────▼──────────┐
        │ compute_jacobian()   │   │ compute_jacobian()  │
        │ K_pos, K_orient      │   │ J_rot[:,5]          │
        │ v_desired            │   │ compute_projected   │
        │                      │   │   _moment_to_base() │
        └──────────┬───────────┘   └──────────┬──────────┘
                   │                          │
        ┌──────────▼───────────┐   ┌──────────▼──────────┐
        │ J_damped_pinv        │   │ compute_base_moment │
        │ v_primary            │   │ _cost()             │
        │                      │   │ U = ½(τ₇-M_proj)²  │
        └──────────┬───────────┘   └──────────┬──────────┘
                   │                          │
                   │                ┌─────────▼──────────┐
                   │                │ compute_null_space │
                   │                │ _torque_correction │
                   │                │ ∆q = -α×N×∇U      │
                   │                └─────────┬──────────┘
                   │                          │
                   └──────────┬───────────────┘
                              │
                    ┌─────────▼──────────┐
                    │   Combine with     │
                    │ correction_scaling │
                    │ v_final = v_primary│
                    │    + scale × ∆q    │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │  Apply velocity    │
                    │  limits & saturate  │
                    └─────────┬──────────┘
                              │
                    ┌─────────▼──────────┐
                    │  data.ctrl[] <--   │
                    │  final velocities  │
                    └────────────────────┘
```

## Moment Projection Visualization

```
ROBOT CONFIGURATION IN SPACE:

                        EE (End-Effector)
                        ───────o
                               /│
                              / │ τ₇ (direct wrist torque)
                             /  │
                            /   │
                      Joint6┤    ├─── Moment M_proj,6 from τ₆
                            \   │
                             \  │
                              \ │
                               \│
                        ────────Base────────
                               
FROM BASE PERSPECTIVE:

    Two moment sources acting on base:
    
    1. DIRECT: τ₇ (wrist rotation) contributes directly
    ───────────────────────────────────────────────────
    
    2. PROJECTED: τ₆ projects through kinematic chain
    ─────────────────────────────────────────────────
    
        M_proj = ||J_rot[:,5]|| × τ₆
        
        where J_rot[:,5] = how joint 6 affects base rotation
                          (depends on current configuration!)
    
    BALANCE: τ₇ = M_proj  →  Zero net moment at base
    ────────────────────────────────────────────────
```

## Cost Function Evolution

```
Time Evolution of Potential Field:

        U (Cost)
        │
     40 │  ╱╲
        │ ╱  ╲    Initial: Large error
     30 │╱    ╲   τ₇ >> M_proj or τ₇ << M_proj
        │      ╲
     20 │       ╲╲
        │        ╲ ╲    Convergence phase
     10 │         ╲ ╲   Gradient descent in action
        │          ╲ ╲╲
      0 │───────────╲─╲─────► Time
        └────────────────────
           0   5  10  15  20
           
Key Events:
• t=0-2s:   High cost, being driven down by gradient
• t=2-8s:   Exponential decay of cost
• t=8-20s:  Plateau at near-zero (well balanced)
```

## Gradient Vector Field

```
MOMENT ERROR (τ₇ - M_proj) SPACE:

    τ₇ - M_proj = 0  ← GOAL (balanced)
    
    ▲
    │
    │ τ₇ > M_proj    │ τ₇ > M_proj
    │ error > 0      │ error > 0
    │ ↓ Reduce τ₇    │ ↓ Reduce τ₇
    │ ↓ Reduce τ₆    │ ↓ Reduce τ₆
    │ ↓↓↓↓↓↓↓↓↓    │ ↓↓↓↓↓↓↓↓↓
    │                │
    ├─────────────────●─────────── τ₇ - M_proj = 0 (GOAL)
    │                │
    │ τ₇ < M_proj    │ τ₇ < M_proj
    │ error < 0      │ error < 0
    │ ↑ Increase τ₇  │ ↑ Increase τ₇
    │ ↑ Increase τ₆  │ ↑ Increase τ₆
    │ ↑↑↑↑↑↑↑↑↑    │ ↑↑↑↑↑↑↑↑↑
    │                │
    
    All arrows point toward goal (balanced moment)
    Arrows indicate gradient descent direction
```

## Null Space Projection

```
7D JOINT SPACE DECOMPOSED:

    ┌─────────────────────────────────────────┐
    │      7D JOINT VELOCITY SPACE            │
    │                                         │
    │  ┌──────────────────────────────────┐  │
    │  │ 6D TASK SPACE (End-Effector)     │  │
    │  │ • Position: 3D                   │  │
    │  │ • Orientation: 3D                │  │
    │  │ Controlled by J·v = v_task       │  │
    │  │                                  │  │
    │  │ v_primary = J⁺ × v_task          │  │
    │  └──────────────────────────────────┘  │
    │           ▲ Task space               ▲
    │           │ (PRIMARY OBJECTIVE)      │
    │           │                          │
    │  ┌────────┴──────────────────────┐   │
    │  │   1D NULL SPACE               │   │
    │  │   (Redundancy)                │   │
    │  │   • No effect on EE task      │   │
    │  │   • Free for secondary goals  │   │
    │  │                               │   │
    │  │   ∆q_null = (I-J⁺J) × ∆q    │   │
    │  │   = Moment correction!       │   │
    │  └───────────────────────────────┘   │
    │           ▲ Null space           ▲
    │           │ (SECONDARY OBJ)      │
    │           │                      │
    │  ┌────────┴────────────────────────┐  │
    │  │ v_final = v_primary + α×∆q_null│  │
    │  │                                 │  │
    │  │ ✓ Task is executed              │  │
    │  │ ✓ Moment is minimized           │  │
    │  │ ✓ No conflict!                  │  │
    │  └─────────────────────────────────┘  │
    │                                         │
    └─────────────────────────────────────────┘
```

## Comparison: Old vs New

```
OLD APPROACH:
─────────────
τ₆ - τ₇ = 0 ?
│
├─ Error: correction ∝ (τ₆ - τ₇)
│
└─ Issues:
   • No physics (arbitrary weighting)
   • Non-smooth (absolute value)
   • Ignores kinematics (J projection)

   Visualization: Simple difference
   ────────────────────────────────
   τ₆ ├─────►
   τ₇ ├─────►
       │balance?
       │
       └─ NO if τ₆ ≠ τ₇


NEW APPROACH:
─────────────
U = ½(τ₇ - M_proj)² → minimize
│
├─ Error: correction ∝ -∇U = -(τ₇ - M_proj) × [J; 1]
│
└─ Advantages:
   ✓ Physics-based (kinematic projection)
   ✓ Smooth (quadratic)
   ✓ Principled optimization
   ✓ Obstacle avoidance analogy

   Visualization: Potential field
   ──────────────────────────────
   τ₇ ├─────────┐
                │ Projected moment
   M_proj ├────►│ from τ₆
                │
                └─ YES if τ₇ = M_proj
```

## Steady-State Behavior

```
STEADY STATE (t >> 5s):
──────────────────────

Primary Task:
├─ Position: XPOS(t) = XREF (tracking achieved)
├─ Orientation: RPOS(t) = RREF (tracking achieved)
└─ Error: Small oscillations only

Secondary Objective:
├─ τ₇: Oscillating around mean value
├─ M_proj: Oscillating in sync with τ₇
├─ Error: |τ₇ - M_proj| ≈ 0 most of the time
└─ Cost: U ≈ 0 (occasionally small spikes)

Moment at Base:
├─ Direct (τ₇): Active
├─ Projected (τ₆ through chain): Approximately cancels τ₇
└─ NET MOMENT: Minimal (structural loading reduced)
```

## Data Monitoring Dashboard

```
REAL-TIME MONITORING:

┌────────────────────────────────────────────────┐
│ PRIMARY TASK (Must maintain)                   │
├────────────────────────────────────────────────┤
│ Position Error: 0.023 m ← Should stay < 0.1   │
│ Orient Error:   0.015 rad ← Should stay < 0.2 │
└────────────────────────────────────────────────┘

┌────────────────────────────────────────────────┐
│ SECONDARY OBJECTIVE (To minimize)              │
├────────────────────────────────────────────────┤
│ Moment Cost U:      0.0045  ← Decreasing ✓   │
│ Moment Error:      +0.0018  ← Oscillates ✓   │
│ τ₇:                 0.234 N·m               │
│ M_proj:             0.232 N·m ← Close to τ₇ ✓ │
└────────────────────────────────────────────────┘

┌────────────────────────────────────────────────┐
│ JOINT STATES                                   │
├────────────────────────────────────────────────┤
│ J6 Torque: 0.0142 N·m                        │
│ J7 Torque: 0.234 N·m                         │
│ J6 Vel:    0.012 rad/s                       │
│ J7 Vel:    0.043 rad/s                       │
└────────────────────────────────────────────────┘

Legend: ✓ = Good, ⚠ = Monitor, ✗ = Adjust tuning
```

## Algorithm Summary (Pseudocode)

```python
while simulating:
    # 1. PRIMARY TASK (High Priority)
    pos_error = target_pos - current_pos
    orient_error = compute_orient_error()
    J = compute_jacobian()
    v_primary = J_damped_pinv @ [pos_error; orient_error]
    
    # 2. MOMENT MINIMIZATION (Low Priority, Null Space)
    tau_6 = current_control[5]
    tau_7 = current_control[6]
    
    J_rot = J[3:6, :]  # Rotational part
    j6_rot = J_rot[:, 5]  # Column for joint 6
    
    # 2a. Compute projected moment (kinematic transmission)
    M_proj = norm(j6_rot) * abs(tau_6)
    
    # 2b. Evaluate cost (potential field)
    error = tau_7 - M_proj
    cost = 0.5 * error^2
    
    # 2c. Compute gradient and correction
    J_pinv = pinv(J)
    N = I - J_pinv @ J  # Null space projector
    correction = -error * gain * [norm(j6_rot); 1]
    v_null = N @ correction
    
    # 3. COMBINE WITH SAFETY
    v_final = v_primary + 0.01 * v_null  # Small scaling for safety
    v_final = clip(v_final, -v_max, v_max)
    
    # 4. APPLY AND STEP
    data.ctrl = v_final
    mujoco.mj_step(model, data)
```

---

**Key Insight**: The algorithm is a hierarchical controller where the primary task (position/orientation) takes precedence, and the moment minimization is a secondary objective that operates in the null space without affecting the primary task performance.
