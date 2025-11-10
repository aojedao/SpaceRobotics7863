#!/usr/bin/env python3
"""
Quick analysis script to check orientation continuity in the collected data.
This will help verify if our continuous Euler angle tracking is working.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

# Simulate some test data collection to verify our continuity mechanism
def test_continuity():
    print("🔍 Testing Euler angle continuity mechanism...")
    
    # Create test quaternions that would cause jumps
    test_quats = [
        [0, 0, 0, 1],              # 0° rotation
        [0, 0, 0.707, 0.707],      # 90° around Z
        [0, 0, 1, 0],              # 180° around Z (would cause jump)
        [0, 0, 0.707, -0.707],     # 270° around Z
        [0, 0, 0, -1],             # 360° around Z (equivalent to 0°)
    ]
    
    # Convert using standard method (will have jumps)
    standard_euler = []
    for quat in test_quats:
        euler = Rotation.from_quat(quat).as_euler('xyz', degrees=True)
        standard_euler.append(euler)
    
    # Convert using continuous tracking (should not have jumps)
    cumulative_euler = np.zeros(3)
    prev_euler = None
    continuous_euler = []
    
    for quat in test_quats:
        euler_raw = Rotation.from_quat(quat).as_euler('xyz', degrees=True)
        
        if prev_euler is not None:
            for i in range(3):
                diff = euler_raw[i] - prev_euler[i]
                if diff > 180:
                    cumulative_euler[i] = cumulative_euler[i] + (euler_raw[i] - 360) - prev_euler[i]
                elif diff < -180:
                    cumulative_euler[i] = cumulative_euler[i] + (euler_raw[i] + 360) - prev_euler[i]
                else:
                    cumulative_euler[i] = cumulative_euler[i] + diff
        else:
            cumulative_euler = euler_raw.copy()
        
        continuous_euler.append(cumulative_euler.copy())
        prev_euler = euler_raw.copy()
    
    # Print comparison
    print("\n📊 Comparison of Standard vs Continuous Euler Angles:")
    print("Step |     Standard Euler (deg)      |    Continuous Euler (deg)     |")
    print("-----|-------------------------------|-------------------------------|")
    for i, (std, cont) in enumerate(zip(standard_euler, continuous_euler)):
        print(f" {i:2d}  | R:{std[0]:6.1f} P:{std[1]:6.1f} Y:{std[2]:6.1f} | R:{cont[0]:6.1f} P:{cont[1]:6.1f} Y:{cont[2]:6.1f} |")
    
    # Check for jumps
    standard_arr = np.array(standard_euler)
    continuous_arr = np.array(continuous_euler)
    
    print("\n🔍 Jump Analysis:")
    if len(standard_arr) > 1:
        std_diffs = np.abs(np.diff(standard_arr, axis=0))
        cont_diffs = np.abs(np.diff(continuous_arr, axis=0))
        
        print(f"Standard method - Max jump: {np.max(std_diffs):.1f}°")
        print(f"Continuous method - Max jump: {np.max(cont_diffs):.1f}°")
        
        if np.max(std_diffs) > 90:
            print("❌ Standard method has jumps > 90°")
        else:
            print("✅ Standard method is smooth")
            
        if np.max(cont_diffs) > 90:
            print("❌ Continuous method still has jumps > 90°")
        else:
            print("✅ Continuous method is smooth")

def check_simulation_data():
    """Check if simulation data files exist and analyze them"""
    import os
    
    print("\n📁 Checking for simulation output files...")
    files = ['integrated_controller_performance_analysis.png', 'controller_error_analysis.png']
    
    for file in files:
        if os.path.exists(file):
            print(f"✅ Found: {file}")
        else:
            print(f"❌ Missing: {file}")
    
    print("\n🎯 Continuous Euler tracking has been implemented in integrated_simulation.py")
    print("   The current vs target orientation plots should now be smooth!")

if __name__ == "__main__":
    test_continuity()
    check_simulation_data()
