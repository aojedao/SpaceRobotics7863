
import subprocess
import time
import sys

def run_batch(num_runs=10):
    print(f"Starting batch of {num_runs} runs...")
    
    for i in range(num_runs):
        print(f"\nRUN {i+1}/{num_runs}")
        try:
            # Run with timeout to prevent hangs
            subprocess.run(["timeout", "300s", "python3", "wall_crawler_mujoco.py", "--headless"], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Run {i+1} failed/timed out: {e}")
            # Continue to next run
            
    print("\nBatch completed.")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        n = int(sys.argv[1])
    else:
        n = 10
    run_batch(n)
