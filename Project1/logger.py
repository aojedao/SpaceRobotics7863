
import json
import os
import datetime

# ============================================================================
# MULTI-RUN SUCCESS TRACKING
# ============================================================================

def load_run_history():
    """Load run history from file"""
    history_file = "run_history.json"
    try:
        with open(history_file, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {'runs': [], 'total': 0, 'successes': 0}

def save_run_history(history):
    """Save run history to file"""
    history_file = "run_history.json"
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)

RECORDING_ENABLED = True

def record_run_result(success: bool, waypoints_reached: int, total_waypoints: int, reason: str = ""):
    """Record the result of a run"""
    if not RECORDING_ENABLED:
        return {'runs': [], 'total': 0, 'successes': 0}

    history = load_run_history()
    
    run_record = {
        'timestamp': datetime.datetime.now().isoformat(),
        'success': success,
        'waypoints_reached': waypoints_reached,
        'total_waypoints': total_waypoints,
        'reason': reason
    }
    
    history['runs'].append(run_record)
    history['total'] += 1
    if success:
        history['successes'] += 1
    
    save_run_history(history)
    return history

def print_success_rate():
    """Print the cumulative success rate across all runs"""
    history = load_run_history()
    
    print(f"\n{'='*60}")
    print("MULTI-RUN SUCCESS RATE")
    print(f"{'='*60}")
    
    if history['total'] == 0:
        print("No runs recorded yet.")
        return
    
    success_rate = (history['successes'] / history['total']) * 100
    print(f"Total Runs: {history['total']}")
    print(f"Successful: {history['successes']}")
    print(f"Failed: {history['total'] - history['successes']}")
    print(f"Success Rate: {success_rate:.1f}%")
    
    # Show last 5 runs
    recent_runs = history['runs'][-5:]
    if recent_runs:
        print(f"\nLast {len(recent_runs)} runs:")
        for i, run in enumerate(recent_runs, 1):
            status = "✅" if run['success'] else "❌"
            wp_info = f"{run['waypoints_reached']}/{run['total_waypoints']}"
            reason = f" ({run['reason']})" if run.get('reason') else ""
            print(f"  {i}. {status} WP: {wp_info}{reason}")
    
    print(f"{'='*60}")
