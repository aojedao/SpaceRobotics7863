# Visualization Scripts for Project 2 Report

This folder contains specialized MuJoCo scripts for generating clean figures and plots for the project report.

## Scripts

### 1. `robot_only_viz.py`
**Purpose**: Spawn only the bimanual robot for clean photography  
**Usage**: `python robot_only_viz.py`

**Features**:
- No environment clutter
- Pre-configured photogenic pose
- Adjustable camera angles
- Perfect for Figure 1 (System Overview)

**Camera Controls**:
- Mouse drag: Rotate view
- Scroll: Zoom
- Right-click drag: Pan
- ESC: Exit

---

### 2. `static_unscrew_scene.py`
**Purpose**: Static scene showing robot aligned with screw  
**Usage**: `python static_unscrew_scene.py`

**Features**:
- Frozen pose (no movement)
- No waypoint markers or spheres
- Clean screw visualization (RED/ORANGE/CYAN)
- Robot positioned for unscrewing task
- Perfect for Figure 2 (Task Setup)

**Configuration**:
- Screw on back wall at (0.0, 1.5, 1.5)
- Left arm aligned for extraction
- Right arm in anchor position

---

### 3. `base_moment_calculator.py`
**Purpose**: Calculate and plot base reaction moments correctly  
**Usage**: Import as module in your simulation

**Key Features**:
- **Correct coordinate transformation**: EE frame → Base frame
- Uses rotation matrices for proper force/torque projection
- Logs moments from both arms separately
- Generates 4-panel analysis plots

**How to Use**:
```python
from visualization.base_moment_calculator import BaseReactionMomentCalculator

# Initialize
calculator = BaseReactionMomentCalculator(model, data)

# In simulation loop
left_force = np.array([Fx, Fy, Fz])   # Force at left gripper (EE frame)
right_force = np.array([Fx, Fy, Fz])  # Force at right gripper (EE frame)
calculator.log_current_state(left_force, right_force)

# After simulation
calculator.print_summary()
calculator.plot_results('base_moments.png')
```

**Output Plots**:
1. Total moment magnitude over time
2. Individual arm contributions
3. Moment components (Mx, My, Mz)
4. Stability assessment with threshold

---

## Known Issues & Fixes

### Issue: Box appears shifted from base location
**Problem**: Visual geometry offset from actual base position  
**Status**: TODO - Need to adjust geom position in XML

**To Fix**:
1. Open `dual_arm_robot.xml`
2. Find `<geom name="base_box" ...>`
3. Adjust `pos` attribute to match base body position
4. Ensure `pos="0 0 0"` relative to parent body

---

### Issue: Incorrect base moment calculation
**Problem**: Previous calculation didn't account for frame transformations  
**Status**: FIXED in `base_moment_calculator.py`

**Solution**:
- Compute rotation matrix from EE frame to base frame
- Transform forces: `F_base = R_base_ee @ F_ee`
- Transform position vectors to base frame
- Use proper cross product: `M = r × F`

---

## Future Enhancements

- [ ] Add ISS module enclosure design
- [ ] Pre-planned trajectory visualization
- [ ] With/without compensation comparison plots
- [ ] Animation export for presentation
- [ ] Multiple camera angle presets

---

## Tips for Report Figures

### Figure 1: System Architecture
- Use `robot_only_viz.py`
- Camera angle: Azimuth=90°, Elevation=-20°
- Show full robot with arms extended
- Add labels/arrows in post-processing

### Figure 2: Task Configuration  
- Use `static_unscrew_scene.py`
- Camera angle: Azimuth=45°, Elevation=-10°
- Focus on EE-screw alignment
- Include color legend

### Figure 3: Performance Plots
- Use `base_moment_calculator.py` output
- 2x2 grid: Moments, Components, Alignment, Trajectory
- Add threshold lines
- Compare with/without compensation

---

## Dependencies

- mujoco (latest)
- numpy
- matplotlib (for plotting)

Install: `pip install mujoco numpy matplotlib`
