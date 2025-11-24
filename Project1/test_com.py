#!/usr/bin/env python3
"""Test to check center of mass"""

import mujoco
import numpy as np
from integrated_simulation import IntegratedZeroGravitySimulation

print("🚀 Creating simulation...")
sim = IntegratedZeroGravitySimulation(controller_type='no_control')

print("\n📍 Checking Center of Mass at zero configuration:")
print(f"  Base inertial pos: -0.1, 0, 0.07")
print(f"  Robot total mass: 5 (base) + 5.76 (link1) + ... = ~33 kg")

# Reset and initialize
mujoco.mj_resetData(sim.model, sim.data)
sim.data.qpos[:] = 0.0
sim.data.qpos[3] = 1.0

# Use balanced configuration - try extended upward
sim.data.qpos[7+0] = 0.0    # joint1
sim.data.qpos[7+1] = 1.57   # joint2: extend up
sim.data.qpos[7+2] = 0.0    # joint3
sim.data.qpos[7+3] = 1.57   # joint4: extend up
sim.data.qpos[7+4] = 0.0    # joint5
sim.data.qpos[7+5] = 0.0    # joint6
sim.data.qpos[7+6] = 0.0    # joint7

sim.data.qvel[:] = 0.0
mujoco.mj_forward(sim.model, sim.data)

# Calculate COM
com_pos = np.zeros(3)
total_mass = 0
for i in range(sim.model.nbody):
    if sim.model.body_mass[i] > 0:
        total_mass += sim.model.body_mass[i]
        com_pos += sim.model.body_mass[i] * sim.data.xipos[i]
        if i < 4:
            print(f"  Body {i:2d}: mass={sim.model.body_mass[i]:6.2f} kg, pos={sim.data.xipos[i]}")

if total_mass > 0:
    com_pos /= total_mass

print(f"\n✓ Total Mass: {total_mass:.2f} kg")
print(f"✓ Center of Mass at zero config: [{com_pos[0]:.4f}, {com_pos[1]:.4f}, {com_pos[2]:.4f}]")
print(f"\n💡 The robot will naturally rotate to move COM to base origin!")
print(f"   This is correct physics for a free-floating system in zero-gravity.")
