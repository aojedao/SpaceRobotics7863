#!/usr/bin/env python3
"""Quick test of no_control controller"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from integrated_simulation import IntegratedZeroGravitySimulation

print("🚀 Testing no_control controller...")
sim = IntegratedZeroGravitySimulation(controller_type='no_control')
print("✓ Controller created successfully")
print(f"✓ Controller type: {type(sim.controller).__name__}")
print(f"✓ Available controllers: {sim.controller.get_available_controllers()}")

# Quick simulation test
import mujoco
print("\n📊 Running 2-second simulation...")
start_time = None
steps = 0
with mujoco.viewer.launch_passive(sim.model, sim.data) as viewer:
    import time
    viewer.cam.lookat[:] = [0.0, 0.5, 1.0]
    viewer.cam.distance = 2.249
    viewer.cam.azimuth = 0.75
    viewer.cam.elevation = -20.0
    
    start = time.time()
    while viewer.is_running() and (time.time() - start) < 2.0:
        mujoco.mj_step(sim.model, sim.data)
        viewer.sync()
        sim.step_controller()
        steps += 1
        
        if steps % 60 == 0:
            print(f"  Step {steps}: Base pos = {sim.data.xpos[0][:3]}")

print(f"✓ Completed {steps} simulation steps")
print("✓ no_control controller test PASSED!")
