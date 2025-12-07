
import numpy as np
import sys
import os

# Ensure we can import the module
sys.path.append(os.getcwd())
from wall_crawler_mujoco import WallCrawlerMuJoCoSimulation

def test_unscrewing():
    print("INITIALIZING UNSCREW TEST...")
    sim = WallCrawlerMuJoCoSimulation("dual_arm_robot.xml")
    
    # Use back wall, moving LEFT to avoid arm crossing
    # Start at 0, Goal at -0.5. 
    # Right arm holds Start (0), Left arm reaches Goal (-0.5).
    start_pos = (0.0, -0.5, 1.0) 
    goal_pos = (-0.5, -0.5, 1.0)
    
    print(f"Setting path: {start_pos} -> {goal_pos}")
    sim.set_start_and_goal(start_pos, goal_pos)
    
    print("Running headless...")
    # Run for enough time to complete 1 step + unscrewing
    success = sim.run_headless(duration=20) 
    
    if success:
        print("\nTEST PASSED: Unscrewing phase completed successfully.")
        sys.exit(0)
    else:
        print("\nTEST FAILED: Did not complete successfully.")
        sys.exit(1)

if __name__ == "__main__":
    test_unscrewing()
