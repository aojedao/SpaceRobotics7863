# Wall Crawler Space Robotics - Project 2 Report Guide

## 📋 Recommended Report Structure (IEEE Double-Column Format)

---

## **1. Introduction & Motivation**

### **A. Problem Statement**

**Key Focus**: Cost-efficiency of robotic manipulation vs. astronaut operations + Importance of torque compensation

**Content to Include**:
- **Challenge in Space Assembly**: Currently, assembly and maintenance tasks on the ISS are performed by astronauts, which is:
  - Extremely costly (astronaut training, EVA preparation)
  - High-risk (exposure, equipment failures)
  - Time-limited (EVA durations)

- **Robotic Manipulation as Solution**:
  - Autonomous or semi-autonomous robotic systems can perform assembly tasks 24/7
  - Significantly reduces cost compared to human astronauts
  - Eliminates human safety risks for routine maintenance

- **Critical Challenge - Base Moment Control**:
  - Dual-arm manipulation creates **moment at the robot base** due to forces at different arm positions
  - Without proper control, reaction moments cause undesired base torques and body rotation
  - **Torque compensation is essential** to maintain base stability and precise manipulation
  - This is especially critical in microgravity where external forces (gravity) cannot stabilize the base

---

### **B. Related Work & Technical Gap**

**Related Work**:
- Dual-arm robotic systems for space assembly (JEMRMS, Canadarm2)
- Wall-climbing and contact maintenance robots
- Dual-manipulator coordination strategies (bipedal walking analogy)

**Technical Gap - Your Innovation**:
- Existing approaches focus on **trajectory tracking** or **contact force control**, but **neglect base moment balance**
- Your approach: Use **dual manipulators as a static walking stability principle**
  - Similar to how bipedal robots maintain stability by balancing forces from both legs
  - Apply same principle to dual arms: maintain base moment equilibrium during manipulation
  
- **Novel Contribution**: 
  - **Jacobian-based torque compensation** (derived from null-space control principles)
  - Same mathematical structure as your obstacle avoidance work: uses **delta H function** to minimize reaction moments
  - Ensures stable base while performing manipulation tasks

---

## **2. Technical Approach & Innovation**

### **A. System Overview**

**Hardware Selection**:
- **KUKA iiwa14 7-DOF Arms**: Chosen because:
  - DLR (German Aerospace Center) uses these arms for assembly tasks both in simulation and on Earth
  - Proven track record for precision manipulation
  - Lightweight design suitable for space applications
  
- **Robotiq 2F85 Gripper**: Selected for:
  - Availability in MuJoCo simulation environment
  - Standard gripper with proven reliability
  - Integrated contact dynamics modeling

**Environment**:
- **Zero-Gravity Simulation**: 
  - MuJoCo physics engine with gravity disabled
  - Simulates free-fall environment (approximates ISS microgravity)
  - Allows focus on dynamics without Earth gravity compensation
  - Contact dynamics and friction modeled realistically

**Control Architecture**:
- **State Machine** with multiple operational modes:
  - Path planning phase
  - Locomotion control phase
  - Manipulation phase (unscrewing with torque compensation)
  - Recovery phase (when arms collide)

---

### **B. Path Planning with Discretization**

**Environment Discretization**:
- Wall surfaces divided into **discrete handle positions** (0.5m step size)
- Each position represents a grip point where the robot can anchor

**Route Selection**:
- **A* Search Algorithm** for optimal path finding
- Considers reach constraints and energy efficiency
- Generates waypoint sequence from start to goal position

**Visualization**:
- Workspace visualization shows reachable regions for each arm
- Allows verification of path feasibility

---

### **C. Locomotion Control**

**Bipedal Walking Analogy**:
- Left and right arms alternate between:
  - **Anchor arm**: Maintains grip on current hold point
  - **Moving arm**: Reaches to next hold point
  
- This alternation maintains continuous contact and stability

**Command Controller**:
- Executes joint-level commands for arm positioning
- Maintains arm poses along the waypoint trajectory

---

### **D. Manipulation with Torque Compensation** ⭐ (Core Innovation)

**Problem**: Manipulation forces at gripper create reaction moments at robot base

**Solution - Torque Compensation Method**:
- **Static Stability Principle**: Similar to bipedal walking, use dual arms to balance forces
- **Jacobian-Based Approach**:
  - Same mathematical structure as obstacle avoidance null-space control
  - Uses **delta H function** to minimize reaction moments at base
  - Optimization objective: Keep base moment near zero while performing manipulation

**Implementation**:
- Compute reaction moments based on gripper forces and distances from base
- Adjust arm joint commands to counterbalance reaction moments
- Maintains stable base during task execution

---

### **E. Screw Extraction Task**

**Task Description**:
- Robot extracts screw from wall after reaching goal position
- J7 (wrist joint) applies torque to rotate screw

**Torque Compensation for Screw Contact**:
- **Constant Torque Model**: Simulates realistic contact dynamics between screw and gripper
- Torque feedback from J7 is continuously monitored
- Extraction continues until screw is fully removed (3 rotations)

**Alignment Before Extraction**:
- **Z-axis angle error metric**: Measures alignment between EE Z-axis and screw axis
- Active alignment phase uses IK control to minimize this error
- Rotation starts when alignment error is minimal (< 15°)
- Ensures clean extraction without binding

---

### **F. Recovery Mechanism**

**Arm Collision Handling**:
- When arms collide or get too close, **recovery controller** engages
- One arm maintains current controller command (stays at current position)
- Other arm adjusts trajectory to avoid collision (turns away from collision point)
- Allows safe continuation of locomotion without jamming

---

## **3. Results and Validation**

### **A. Experiment Setup**

**Simulation Environment**:
- MuJoCo physics engine with ISS module bounds
- Planar walls (back, front walls) as primary test surfaces
- Zero-gravity with realistic contact dynamics

**Test Scenarios**:
- Simple wall traversal: Start position → Goal position
- Multi-waypoint paths: 4-6 waypoints across continuous wall surface
- Unscrewing task: Screw extraction after reaching goal

---

### **B. Key Performance Metrics**

| Metric | Description |
|--------|-------------|
| **Waypoint Spacing** | 0.5m between discrete handle positions |
| **Z-axis Alignment Error** | Angle between EE Z-axis and screw axis (minimized during task) |
| **Base Moment Stability** | Reaction moment at base during manipulation |
| **Screw Extraction Completion** | 3 full rotations to extract screw |
| **Arm Collision Avoidance** | Recovery success rate during adjacent arm movements |

---

### **C. Visualization Elements**

**In-Simulation Visualization**:

1. **Screw Representation** (Color-coded for clarity):
   - **RED Head**: Large visible screw head (6cm)
   - **ORANGE Shaft**: Main body of screw
   - **CYAN Thread**: Thread geometry
   - **MAGENTA Marker Sphere**: Auxiliary visualization at screw location

2. **Contact Points** (During Manipulation):
   - **YELLOW**: Gripper contact with screw
   - **GREEN**: Anchor arm grip point
   - **BLUE**: Trajectory/target positions

3. **Workspace Visualization**:
   - Arm reachable regions shown as transparent spheres
   - Available handle positions marked along wall
   - Planned trajectory highlighted with waypoint markers

---

### **D. Suggested Report Figures**

**Figure 1: System Architecture & Control Flow**
```
┌─────────────────────────────────────────┐
│   Dual-Arm Wall Crawler System          │
├──────────────┬──────────────┬───────────┤
│              │              │           │
│ Path Planner │  Locomotion  │Torque     │
│ (A* search)  │ (Dual arms)  │Comp.      │
│              │              │(Manip.)   │
└──────────────┴──────────────┴───────────┘
        ↓           ↓              ↓
┌─────────────────────────────────────────┐
│  MuJoCo Physics (Zero Gravity)          │
│  [Base] [L-Arm] [R-Arm] [Screw]        │
└─────────────────────────────────────────┘
   State Machine for Control Modes
   ↓
  [Path Planning] → [Locomotion] → [Manipulation] → [Recovery]
```

**Figure 2: Three-Panel Simulation Snapshots**
- **Panel A**: Locomotion phase (arm approaching wall with visible workspace)
- **Panel B**: Alignment phase (EE approaching screw, Z-axis alignment indicator)
- **Panel C**: Extraction phase (J7 rotating, screw with color-coded geometry, contact points)
- Include legend: RED=screw, YELLOW=contacts, GREEN=anchor, BLUE=targets

**Figure 3: Dual-Panel Performance Results**
```
┌──────────────────────────┬──────────────────────┐
│ Z-axis Alignment Error   │ Base Reaction Moment │
│ (degrees vs time)        │ (N·m vs time)        │
│ Shows error → 0 phase    │ Shows stabilization  │
│ during alignment & task  │ by torque comp.      │
└──────────────────────────┼──────────────────────┘
│ Screw Extraction Progress│ Arm Positions        │
│ (rotation angle vs time) │ (xyz coordinates)    │
│ Shows 3 rotation cycles  │ Shows trajectory     │
│ with constant torque     │ following & recovery │
└──────────────────────────┴──────────────────────┘
```

---

### **E. Results Discussion - What to Present**

**1. Path Planning & Locomotion**:
- "The A* planner successfully discretizes the wall into 0.5m handle positions and generates collision-free paths"
- "Dual-arm alternation maintains continuous contact during traversal"
- Show successful navigation through {X} waypoints

**2. Alignment Control Performance**:
- "Z-axis alignment error is minimized to < 15° before screw extraction begins"
- "Active IK control converges within {X} seconds"
- Present alignment error plot showing convergence

**3. Torque Compensation (Key Innovation)**:
- "Torque compensation successfully maintains base moment near zero during manipulation"
- "Reaction moments reduced by {X}% compared to uncompensated approach"
- Compare base stability: "With compensation: stable base | Without: significant base drift"

**4. Screw Extraction Task**:
- "Constant contact torque model enables realistic screw extraction"
- "Robot completes 3 full rotations within contact force limits"
- Show J7 torque feedback plot and screw displacement

**5. Recovery Mechanism**:
- "When arm collisions detected, recovery controller successfully avoids jamming"
- "One arm holds position while other arm steers around collision"
- Present example of collision avoidance sequence

**6. Overall Mission Success**:
- "System successfully demonstrates complete mission: Locomotion → Alignment → Manipulation"
- Quantify success rate across multiple test runs

---

## **4. Key Innovation Summary for Report**

Focus on **three main contributions**:

1. **Jacobian-Based Torque Compensation**
   - Applies null-space control principles to minimize base reaction moments
   - Uses delta H function (same as obstacle avoidance work) for optimization
   - Enables stable manipulation in zero-gravity

2. **Static Stability via Dual Manipulators**
   - Adapts bipedal walking stability principles to dual-arm systems
   - Leverages two arms as "legs" to maintain equilibrium
   - Ensures base stability while performing manipulation

3. **Integrated Locomotion-Manipulation System**
   - Single platform handles both wall-crawling and manipulation tasks
   - Active alignment phase before contact tasks
   - Recovery mechanism for arm collision avoidance

---

## **5. Documentation Sources for Your Report**

**Primary References from Your Work**:
- `wall_crawler_mujoco.py`: Control algorithms, state machine, torque compensation implementation
- `dual_arm_robot.xml`: System model and specifications
- `COPILOT_CONTEXT.md`: Technical parameters and control gains
- Generated plots: alignment errors, base moments, trajectory tracking

**External References** (for Related Work section):
- DLR KUKA iiwa14 documentation
- MuJoCo physics engine papers
- Null-space control and Jacobian-based methods (cite your obstacle avoidance work)
- Dual-manipulator coordination literature
- ISS assembly and maintenance procedures

---

## **6. Writing Tips for IEEE Format**

1. **Start each section with motivation** - "Why does this matter?"
2. **Use figures early and often** - Readers understand visuals faster
3. **Connect to your innovation** - Always relate back to torque compensation and static stability
4. **Quantify results** - Avoid vague statements; provide numbers
5. **Be specific about what's novel** - Clearly distinguish your contribution from prior work

---

## **7. Section Word Count Guidance** (for 2-4 page report)

- **Introduction**: 300-400 words
- **Technical Approach**: 500-700 words (biggest section)
- **Results**: 300-500 words
- **Conclusion/Future Work**: 100-200 words
- **References**: Not counted in page limit

**Total Body**: ~1500-1900 words (fits 2-3 pages in IEEE double-column format)
