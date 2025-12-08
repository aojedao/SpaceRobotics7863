#!/usr/bin/env python3
"""
Test runner for the wall-crawler simulation.
Runs multiple tests with different start/goal positions and reports success rate.
"""

import subprocess
import sys
import os
import time
import json

def parse_success_rate_from_json():
    """Parse the success rate from the simulation's JSON log file."""
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    json_file = os.path.join(script_dir, "run_history.json")
    
    if not os.path.exists(json_file):
        return None, None, None, []
    
    try:
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        runs = data.get('runs', [])
        total = len(runs)
        success = sum(1 for r in runs if r.get('success', False))
        
        if total > 0:
            rate = (success / total) * 100
            return total, success, rate, runs
        return None, None, None, []
        
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return None, None, None, []

def main():
    print("=" * 60)
    print("WALL-CRAWLER SIMULATION TEST REPORT")
    print("=" * 60)
    
    # Check current success rate from JSON log
    total, success, rate, runs = parse_success_rate_from_json()
    
    if rate is not None:
        print(f"\nHistorical Performance (from run_history.json):")
        print(f"  Total Runs: {total}")
        print(f"  Successful: {success}")
        print(f"  Failed: {total - success}")
        print(f"  Success Rate: {rate:.1f}%")
        
        # Show last 10 runs
        print(f"\nLast 10 runs:")
        for i, run in enumerate(runs[-10:], 1):
            status = "✅" if run.get('success') else "❌"
            wp = run.get('waypoints_reached', 0)
            total_wp = run.get('total_waypoints', 0)
            reason = run.get('reason', 'unknown')
            print(f"  {i:2d}. {status} WP: {wp}/{total_wp} ({reason})")
        
        # Analyze recent trends (last 20 runs)
        recent = runs[-20:] if len(runs) >= 20 else runs
        recent_success = sum(1 for r in recent if r.get('success', False))
        recent_rate = (recent_success / len(recent)) * 100 if recent else 0
        
        print(f"\nRecent Performance (last {len(recent)} runs):")
        print(f"  Recent Success Rate: {recent_rate:.1f}%")
        
        # Overall assessment
        print(f"\nPerformance Assessment:")
        if rate >= 80:
            print(f"  ✓ EXCELLENT - {rate:.1f}% overall success rate")
        elif rate >= 60:
            print(f"  ○ GOOD - {rate:.1f}% overall success rate")
        elif rate >= 40:
            print(f"  △ MODERATE - {rate:.1f}% overall success rate")
        else:
            print(f"  ✗ NEEDS IMPROVEMENT - {rate:.1f}% overall success rate")
            
        # Common failure points
        failures = [r for r in runs if not r.get('success')]
        if failures:
            failure_wps = {}
            for f in failures:
                reason = f.get('reason', 'unknown')
                failure_wps[reason] = failure_wps.get(reason, 0) + 1
            
            print(f"\nMost Common Failure Points:")
            sorted_failures = sorted(failure_wps.items(), key=lambda x: x[1], reverse=True)
            for reason, count in sorted_failures[:5]:
                print(f"  - {reason}: {count} times ({count/len(failures)*100:.0f}%)")
                
    else:
        print("\nNo historical data found. Run the simulation to generate data.")
    
    print("=" * 60)
    
    return 0 if (rate and rate >= 50) else 1

if __name__ == "__main__":
    sys.exit(main())
