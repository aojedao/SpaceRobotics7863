#!/usr/bin/env python3
"""
Run multiple simulation tests and report success rate.
Uses headless mode to run simulations without GUI.
"""

import subprocess
import sys
import os
import json
import time
from datetime import datetime

def run_single_test(test_num, timeout=120):
    """Run a single simulation test with timeout."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, "wall_crawler_mujoco.py")
    
    print(f"\n{'='*60}")
    print(f"TEST {test_num}: Starting...")
    print(f"{'='*60}")
    
    start_time = time.time()
    
    try:
        # Run the simulation - it will auto-record results to run_history.json
        result = subprocess.run(
            ["conda", "run", "-n", "space-robotics", "--no-capture-output",
             "python", script_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=script_dir,
            env={**os.environ, 'DISPLAY': ''}  # Disable display
        )
        
        elapsed = time.time() - start_time
        output = result.stdout + result.stderr
        
        # Check for success indicators
        if "✅ MISSION COMPLETE" in output or "completed" in output.lower():
            print(f"TEST {test_num}: ✅ SUCCESS in {elapsed:.1f}s")
            return True
        else:
            print(f"TEST {test_num}: ❌ FAILED in {elapsed:.1f}s")
            # Extract failure reason
            for line in output.split('\n'):
                if 'stopped at' in line.lower() or 'failed' in line.lower():
                    print(f"  Reason: {line.strip()}")
                    break
            return False
            
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"TEST {test_num}: ⏱️ TIMEOUT after {elapsed:.1f}s")
        return False
    except Exception as e:
        print(f"TEST {test_num}: ❌ ERROR: {e}")
        return False

def get_current_stats():
    """Get current success rate from run_history.json"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    history_file = os.path.join(script_dir, "run_history.json")
    
    if not os.path.exists(history_file):
        return 0, 0, 0
    
    try:
        with open(history_file, 'r') as f:
            data = json.load(f)
        
        runs = data.get('runs', [])
        total = len(runs)
        success = sum(1 for r in runs if r.get('success', False))
        rate = (success / total * 100) if total > 0 else 0
        
        return total, success, rate
    except:
        return 0, 0, 0

def main():
    num_tests = 5  # Default number of tests
    
    if len(sys.argv) > 1:
        try:
            num_tests = int(sys.argv[1])
        except ValueError:
            print(f"Usage: {sys.argv[0]} [num_tests]")
            return 1
    
    print("=" * 60)
    print(f"MULTI-RUN SIMULATION TEST - {num_tests} runs")
    print("=" * 60)
    print(f"Arms: Side-mounted (X-axis ends, pointing outward)")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Get initial stats
    init_total, init_success, _ = get_current_stats()
    
    # Run tests
    results = []
    for i in range(1, num_tests + 1):
        success = run_single_test(i)
        results.append(success)
    
    # Get final stats
    final_total, final_success, final_rate = get_current_stats()
    
    # Calculate this session's stats
    session_total = final_total - init_total
    session_success = final_success - init_success
    session_rate = (session_success / session_total * 100) if session_total > 0 else 0
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    print(f"\nThis Session ({session_total} tests):")
    print(f"  Successful: {session_success}")
    print(f"  Failed: {session_total - session_success}")
    print(f"  Success Rate: {session_rate:.1f}%")
    
    print(f"\nOverall (all time, {final_total} tests):")
    print(f"  Successful: {final_success}")
    print(f"  Failed: {final_total - final_success}")
    print(f"  Success Rate: {final_rate:.1f}%")
    
    # Test breakdown
    print(f"\nTest Results:")
    for i, success in enumerate(results, 1):
        status = "✅" if success else "❌"
        print(f"  Test {i}: {status}")
    
    print("=" * 60)
    
    return 0 if session_rate >= 50 else 1

if __name__ == "__main__":
    sys.exit(main())
