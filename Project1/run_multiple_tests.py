#!/usr/bin/env python3
"""
Run multiple simulation tests IN PARALLEL and report success rate.
Uses headless mode and disables history recording to ensure concurrency safety.
"""

import subprocess
import sys
import os
import time
from datetime import datetime
import concurrent.futures
import threading

# Lock for printing to avoid garbled output
print_lock = threading.Lock()

def run_single_test_parallel(test_num, timeout=600, visualize=False):
    """Run a single simulation test with timeout in a separate process."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    script_path = os.path.join(script_dir, "wall_crawler_mujoco.py")
    
    with print_lock:
        print(f"TEST {test_num}: 🚀 STARTED")
    
    start_time = time.time()
    
    try:
        # Construct command
        # Revert to conda run to ensure 'mujoco' module is found
        cmd = ["conda", "run", "-n", "space-robotics", "python", script_path]
        
        # Args
        env = os.environ.copy()
        if not visualize:
            cmd.append("--no-record")
            cmd.append("--headless")  # Use headless mode for parallel testing
            env['MUJOCO_GL'] = 'osmesa'  # Use OSMesa for off-screen rendering
            capture = True
        else:
            # If visualizing, we WANT to see output and GUI
            # cmd does NOT include --no-record (so it keeps default behavior if any)
            # But wait, run_multiple_tests suppresses history recording usually.
            # If we visualize, we probably still want to avoid corrupting history if we run multiple?
            # Actually, standard run records history. If we run sequential, it's fine.
            cmd.append("--no-record") # Still disable recording to keep stats clean/controlled by this script? 
            # User wants to SEE it.
            capture = False # Let it print to stdout
            
        # Run process
        if capture:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=script_dir,
                env=env
            )
            output = result.stdout + result.stderr
        else:
            # For visualization, just run it and let user watch.
            # We can't easily capture output AND show it in real time without complexity.
            # But we need to know if it successed.
            # wall_crawler_mujoco.py prints "MISSION COMPLETE".
            # We can pipe stdout to a file and read it, or use Popen.
            # Simple approach: Run it, wait. Return 'Unknown' if we can't parse output easily, 
            # OR use subprocess.PIPE and read line by line (complex).
            # Alternative: Assume success if return code 0? No, simulation might exit 0 but fail task.
            # Let's use capture_output=True but print it? No, that hides GUI logs.
            # Best: just Use subprocess.run with capture_output=True, but remove invalid env for Display.
            # The GUI will pop up, logs captured.
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=script_dir,
                env=env
            )
            output = result.stdout + result.stderr
        
        elapsed = time.time() - start_time
        
        # Check for success indicators
        success = False
        reason = "Unknown failure"
        
        if "✅ GLOBAL_SUCCESS_TOKEN" in output:
            success = True
            status_icon = "✅"
        else:
            status_icon = "❌"
            # Extract failure reason and progress
            reason = "Unknown failure"
            progress = "0/?"
            
            # Look for progress (WP reached)
            # Pattern: "WPX REACHED"
            import re
            wp_matches = re.findall(r"WP(\d+) REACHED", output)
            if wp_matches:
                reached = [int(x) for x in wp_matches]
                max_reached = max(reached) if reached else 0
                progress = f"WP{max_reached}"
            
            # Look for specific failure messages
            if "❌ MAX RECOVERY ATTEMPTS" in output:
                reason = "Max Recovery Attempts Reached"
            elif "⏱️ TIMEOUT" in output: # From this script, or if sim prints it
                pass # Already handled by exception block mostly, but check output
            elif "still reaching" in output.lower():
                reason = "Stuck / Timeout reaching waypoint"
            
            reason = f"{reason} (Ended at {progress})"
        
        with print_lock:
            if success:
                print(f"TEST {test_num}: {status_icon} SUCCESS in {elapsed:.1f}s")
            else:
                print(f"TEST {test_num}: {status_icon} FAILED in {elapsed:.1f}s ({reason})")
                # Print last few lines of output for diagnosis
                print( "    DEBUG OUTPUT:")
                lines = output.split('\n')
                for line in lines[-20:]:
                    print(f"    {line}")
                
        return success, elapsed, reason
            
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        with print_lock:
            print(f"TEST {test_num}: ⏱️ TIMEOUT after {elapsed:.1f}s")
        return False, elapsed, "Timeout"
    except Exception as e:
        with print_lock:
            print(f"TEST {test_num}: 💥 ERROR: {e}")
        return False, 0, str(e)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run multiple simulation tests')
    parser.add_argument('num_tests', type=int, nargs='?', default=5, help='Number of tests to run')
    parser.add_argument('workers', type=int, nargs='?', default=4, help='Number of parallel workers')
    parser.add_argument('--visualize', '-v', action='store_true', help='Enable visualization (runs sequentially)')
    args = parser.parse_args()
    
    num_tests = args.num_tests
    max_workers = args.workers
    visualize = args.visualize
    
    if visualize:
        print("📺 VISUALIZATION ENABLED: Launching parallel instances")
        # For visualization, we allow parallel workers (user requested 5 on screen)
        # max_workers is respected.
    
    print("=" * 60)
    print(f"PARALLEL SIMULATION TEST - {num_tests} runs")
    print(f"Workers: {max_workers}")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    start_total = time.time()
    results = []
    
    # If visualization is enabled, run sequentially (can't have multiple GUIs)
    if visualize:
        print("⚠️  Visualization enabled: Running tests SEQUENTIALLY")
        for i in range(num_tests):
            test_num = i + 1
            try:
                success, elapsed, reason = run_single_test_parallel(test_num, 120, visualize)
                results.append((test_num, success, elapsed, reason))
            except Exception as exc:
                print(f"Test {test_num} generated an exception: {exc}")
                results.append((test_num, False, 0, str(exc)))
            # Small delay between tests
            time.sleep(0.5)
    else:
        # Run tests in parallel (headless mode)
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_test = {executor.submit(run_single_test_parallel, i+1, 600, visualize): i+1 for i in range(num_tests)}
            
            for future in concurrent.futures.as_completed(future_to_test):
                test_num = future_to_test[future]
                try:
                    success, elapsed, reason = future.result()
                    results.append((test_num, success, elapsed, reason))
                except Exception as exc:
                    print(f"Test {test_num} generated an exception: {exc}")
                    results.append((test_num, False, 0, str(exc)))

    total_time = time.time() - start_total
    
    # Sort results by test number
    results.sort(key=lambda x: x[0])
    
    success_count = sum(1 for _, s, _, _ in results if s)
    success_rate = (success_count / num_tests) * 100
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    print(f"Total Time: {total_time:.1f}s")
    print(f"Successful: {success_count}/{num_tests}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    print("\nBreakdown:")
    for test_num, success, elapsed, reason in results:
        status = "✅" if success else "❌"
        print(f"  Test {test_num}: {status} ({elapsed:.1f}s) - {reason}")
    
    print("=" * 60)
    
    return 0 if success_rate >= 70 else 1

if __name__ == "__main__":
    sys.exit(main())
