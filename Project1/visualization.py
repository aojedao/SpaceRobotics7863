"""
Visualization Module for Wall Crawler Robot

This module contains all visualization, plotting, and terminal output
functions for the dual-arm wall-crawling robot simulation.

Classes:
    - SimulationLogger: Terminal output and status printing
    - PlotGenerator: Matplotlib plotting for analysis graphs
    - SceneRenderer: MuJoCo scene geometry rendering
"""

import numpy as np
import mujoco
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
import time


# ============================================================================
# SIMULATION LOGGER - Terminal Output
# ============================================================================

class SimulationLogger:
    """Handles all terminal printing and status output for the simulation."""
    
    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self._last_status_time = 0
        self._status_interval = 0.1  # Minimum seconds between status updates
    
    def print_header(self, title: str, width: int = 60):
        """Print a section header"""
        print(f"\n{'='*width}")
        print(title)
        print(f"{'='*width}")
    
    def print_subheader(self, title: str, width: int = 60):
        """Print a subsection header"""
        print(f"\n{'-'*width}")
        print(title)
        print(f"{'-'*width}")
    
    def print_initialization(self, model_path: str, step_size: float, 
                            arm_reach: float, effective_reach: float,
                            total_positions: int, wall_counts: Dict[str, int]):
        """Print initialization information"""
        self.print_header("MUJOCO WALL-CRAWLER SIMULATION")
        print(f"Loading model: {model_path}")
        print("✓ Model loaded successfully!")
        
        self.print_header("WALL POSITION MAPPER INITIALIZED")
        print(f"Step size: {step_size}m")
        print(f"Arm reach: {arm_reach}m")
        print(f"Effective reach for planning: {effective_reach}m")
        print(f"Total wall positions: {total_positions}")
        for wall, count in wall_counts.items():
            print(f"  - {wall.capitalize()}: {count}")
    
    def print_path_planning(self, start_pos: Tuple, goal_pos: Tuple,
                           snapped_start: Tuple, snapped_goal: Tuple,
                           start_wall: str, goal_wall: str):
        """Print path planning information"""
        self.print_header("PATH PLANNING")
        print(f"Requested Start: {start_pos} → Snapped to: {snapped_start} ({start_wall})")
        print(f"Requested Goal:  {goal_pos} → Snapped to: {snapped_goal} ({goal_wall})")
    
    def print_path_found(self, path: List, total_steps: int):
        """Print path details when found"""
        print(f"\n✓ Path found with {total_steps} steps:")
        for i, state in enumerate(path):
            print(f"  Step {i+1}: {state.wall:10s} | "
                  f"({state.position[0]:6.2f}, {state.position[1]:6.2f}, {state.position[2]:6.2f}) | "
                  f"{state.active_arm.upper()} arm")
    
    def print_no_path_found(self):
        """Print message when no path is found"""
        print("✗ No path found!")
    
    def print_state_transition(self, old_state: str, new_state: str):
        """Print state machine transition"""
        print(f"State: {old_state} → {new_state}")
    
    def print_visualization_legend(self, workspace_radius: float):
        """Print visualization legend for MuJoCo viewer"""
        self.print_header("STARTING MUJOCO VISUALIZATION")
        print("Visualization Legend:")
        print("  🔘 Gray spheres: Wall grip positions")
        print("  🟢 Green sphere: Start position / Completed waypoints")
        print("  🔴 Red sphere: Goal position")
        print("  🔵 Blue spheres: Upcoming waypoints")
        print("  🟡 Yellow sphere: Current target waypoint")
        print("  🟠 Orange sphere: Active arm target")
        print("  ⎯⎯ Cyan lines: Planned route")
        print("  ⎯⎯ Green lines: Completed route segments")
        print(f"  🔵 Blue transparent sphere: Left arm workspace ({workspace_radius}m from elbow)")
        print(f"  🟠 Orange transparent sphere: Right arm workspace ({workspace_radius}m from elbow)")
        print("="*60)
        print("Controls:")
        print("  - Mouse drag: Rotate camera")
        print("  - Scroll: Zoom")
        print("  - Double-click: Track object")
        print("  - ESC: Exit")
        print("="*60)
        print("⚠️  Robot is FREE-FLOATING (physics enabled)")
        print("="*60)
    
    def print_locomotion_init(self, wp1: np.ndarray, body_pos: Tuple):
        """Print locomotion initialization info"""
        print(f"\n📍 WALL-CRAWLER LOCOMOTION TEST")
        print(f"  First anchor (WP1): ({wp1[0]:.2f}, {wp1[1]:.2f}, {wp1[2]:.2f})")
        print(f"  Body position: ({body_pos[0]:.2f}, {body_pos[1]:.2f}, {body_pos[2]:.2f})")
    
    def print_trajectory_waypoints(self, waypoints: List, walls: List, arms: List):
        """Print full trajectory waypoints"""
        total = len(waypoints)
        print(f"\n📍 FULL TRAJECTORY: {total} waypoints")
        for i, (wp, wall, arm) in enumerate(zip(waypoints, walls, arms)):
            is_final = " [FINAL GOAL]" if i == total - 1 else ""
            print(f"  WP{i+1}: ({wp[0]:.2f}, {wp[1]:.2f}, {wp[2]:.2f}) | {wall:12s} | {arm.upper()} arm{is_final}")
    
    def print_phase_status(self, phase_timer: int, arm: str, waypoint_idx: int,
                          error: float, body_pos: np.ndarray, 
                          is_final: bool = False, extra: str = ""):
        """Print periodic phase status update"""
        suffix = " [FINAL]" if is_final else ""
        print(f"  [{phase_timer:4d}] {arm.upper()}→WP{waypoint_idx+1} | "
              f"Error: {error:.3f}m | Body: ({body_pos[0]:.2f}, {body_pos[1]:.2f}, {body_pos[2]:.2f}){suffix}{extra}")
    
    def print_waypoint_reached(self, waypoint_idx: int, arm: str, error: float):
        """Print waypoint reached message"""
        print(f"\n✅ WP{waypoint_idx+1} REACHED by {arm.upper()} arm! Error: {error:.3f}m")
    
    def print_anchor_set(self, arm: str, position: np.ndarray):
        """Print anchor set message"""
        print(f"  🔒 {arm.upper()} arm ANCHORED at sphere ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})")
    
    def print_anchor_released(self, arm: str):
        """Print anchor released message"""
        print(f"  🔓 {arm.upper()} arm RELEASED")
    
    def print_recovery_start(self, attempts: int, max_attempts: int,
                            current_j1: float, target_j1: float,
                            strategy: str, time_stuck: float, 
                            error: float, best_error: float):
        """Print stuck recovery start message"""
        print(f"\n⚠️ STUCK DETECTED after {time_stuck:.1f}s without progress")
        print(f"  Current error: {error:.3f}m | Best seen: {best_error:.3f}m")
        print(f"  Starting RECOVERY {attempts}/{max_attempts}:")
        print(f"    - {strategy}")
        print(f"    - J1: {np.degrees(current_j1):.1f}° → {np.degrees(target_j1):.1f}°")
    
    def print_recovery_progress(self, attempt: int, max_attempts: int,
                               current_j1: float, target_j1: float,
                               time_remaining: float):
        """Print recovery progress"""
        print(f"  🔄 RECOVERY [{attempt}/{max_attempts}] "
              f"J1: {np.degrees(current_j1):.1f}° → {np.degrees(target_j1):.1f}° | "
              f"{time_remaining:.1f}s left")
    
    def print_recovery_complete(self, attempt: int):
        """Print recovery complete message"""
        print(f"\n  ✅ Recovery {attempt} complete - J1 rotated, resuming control")
    
    def print_trajectory_complete(self, wall: str, body_pos: np.ndarray):
        """Print trajectory complete message"""
        print(f"\n🎉 TRAJECTORY COMPLETE!")
        print(f"  Final position on {wall} wall")
        print(f"  Body at: ({body_pos[0]:.2f}, {body_pos[1]:.2f}, {body_pos[2]:.2f})")
    
    def print_screw_spawned(self, position: np.ndarray, wall: str, arm: str):
        """Print screw spawn message"""
        print(f"\n  🔩 SCREW SPAWNED at ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})")
        print(f"     Wall: {wall}")
        print(f"     Starting screw rotation with {arm.upper()} arm joint7...")
    
    def print_screw_rotation(self, rotations: float, total_rotations: float):
        """Print screw rotation progress"""
        print(f"  🔩 Screw rotation: {rotations:.2f}/{total_rotations} turns")
    
    def print_screw_complete(self, rotations: float, elapsed: float):
        """Print screw task complete"""
        print(f"\n  ✅ SCREW TASK COMPLETE!")
        print(f"     {rotations} rotations completed in {elapsed:.1f}s")
    
    def print_summary(self, success: bool, waypoints_reached: int, 
                     total_waypoints: int, anchor_history: Dict):
        """Print final summary"""
        self.print_header("TRAJECTORY EXECUTION SUMMARY")
        
        if success:
            print("✅ TRAJECTORY COMPLETED SUCCESSFULLY!")
        else:
            print("❌ TRAJECTORY DID NOT COMPLETE")
            print(f"   Reached waypoint {waypoints_reached} of {total_waypoints}")
        
        # Anchor deviation analysis
        self.print_header("ANCHOR DEVIATION ANALYSIS")
        print(f"{'WP':>4} | {'Arm':>6} | {'Max Dev (m)':>12} | {'Avg Dev (m)':>12} | {'Samples':>8}")
        print("-" * 60)
        
        for wp_idx in sorted(anchor_history.keys()):
            data = anchor_history[wp_idx]
            arm = data['arm']
            max_dev = data['max_deviation']
            deviations = data['deviations']
            avg_dev = np.mean(deviations) if deviations else 0.0
            n_samples = len(deviations)
            
            print(f"{wp_idx+1:>4} | {arm.upper():>6} | {max_dev:>12.4f} | {avg_dev:>12.4f} | {n_samples:>8}")
        
        print("-" * 60)
    
    def print_success_rate(self, successes: int, total: int):
        """Print cumulative success rate"""
        if total > 0:
            rate = (successes / total) * 100
            print(f"\n📊 CUMULATIVE SUCCESS RATE: {successes}/{total} ({rate:.1f}%)")


# ============================================================================
# PLOT GENERATOR - Matplotlib Graphs
# ============================================================================

class PlotGenerator:
    """Generates matplotlib plots for simulation analysis."""
    
    def __init__(self, plot_timeout: float = 5.0):
        """
        Initialize plot generator.
        
        Args:
            plot_timeout: Time in seconds to display interactive plots
        """
        self.plot_timeout = plot_timeout
    
    def _setup_matplotlib(self):
        """Setup matplotlib with TkAgg backend"""
        import matplotlib
        try:
            matplotlib.use('TkAgg')
        except:
            pass
        import matplotlib.pyplot as plt
        return plt
    
    def generate_anchor_deviation_plot(self, anchor_history: Dict, 
                                       success: bool,
                                       save_path: str = "anchor_deviation_analysis.png"):
        """Generate anchor deviation analysis plot.
        
        Args:
            anchor_history: Dictionary of anchor deviation data per waypoint
            success: Whether trajectory completed successfully
            save_path: Path to save the plot
        """
        if not anchor_history:
            print("⚠️ No anchor deviation data to plot")
            return
        
        try:
            plt = self._setup_matplotlib()
            
            max_deviations = []
            wp_labels = []
            
            for wp_idx in sorted(anchor_history.keys()):
                data = anchor_history[wp_idx]
                max_deviations.append(data['max_deviation'])
                wp_labels.append(f"WP{wp_idx+1}\n({data['arm'][0].upper()})")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Bar chart of max deviations
            colors = ['#2196F3' if 'L' in label else '#FF9800' for label in wp_labels]
            bars = ax1.bar(range(len(max_deviations)), max_deviations, 
                          color=colors, edgecolor='black', linewidth=1.2)
            ax1.set_xticks(range(len(max_deviations)))
            ax1.set_xticklabels(wp_labels)
            ax1.set_xlabel('Waypoint (Arm)', fontsize=12)
            ax1.set_ylabel('Max End-Effector Deviation (m)', fontsize=12)
            ax1.set_title('Maximum EE Deviation from Anchor Point per Waypoint', 
                         fontsize=14, fontweight='bold')
            ax1.axhline(y=0.05, color='green', linestyle='--', linewidth=1.5, label='Good (<5cm)')
            ax1.axhline(y=0.10, color='orange', linestyle='--', linewidth=1.5, label='Acceptable (<10cm)')
            ax1.axhline(y=0.20, color='red', linestyle='--', linewidth=1.5, label='Limit (20cm)')
            ax1.legend(loc='upper right')
            ax1.set_ylim(0, max(0.25, max(max_deviations) * 1.2))
            ax1.grid(axis='y', alpha=0.3)
            
            # Add value labels on bars
            for bar, val in zip(bars, max_deviations):
                height = bar.get_height()
                ax1.annotate(f'{val:.3f}m',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=9)
            
            # Time series of all deviations
            all_deviations = []
            all_wp_indices = []
            for wp_idx in sorted(anchor_history.keys()):
                devs = anchor_history[wp_idx]['deviations']
                all_deviations.extend(devs)
                all_wp_indices.extend([wp_idx] * len(devs))
            
            if all_deviations:
                ax2.scatter(range(len(all_deviations)), all_deviations, 
                           c=all_wp_indices, cmap='viridis', alpha=0.5, s=2)
                ax2.set_xlabel('Time Step (during anchoring)', fontsize=12)
                ax2.set_ylabel('EE Deviation from Anchor (m)', fontsize=12)
                ax2.set_title('End-Effector Deviation Over Time (All Anchors)', 
                             fontsize=14, fontweight='bold')
                ax2.axhline(y=0.05, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
                ax2.axhline(y=0.10, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
                ax2.set_ylim(0, max(0.15, max(all_deviations) * 1.2))
                ax2.grid(alpha=0.3)
                
                sm = plt.cm.ScalarMappable(cmap='viridis', 
                                          norm=plt.Normalize(vmin=min(all_wp_indices), 
                                                            vmax=max(all_wp_indices)))
                sm.set_array([])
                cbar = plt.colorbar(sm, ax=ax2)
                cbar.set_label('Waypoint Index')
            
            status = 'SUCCESS' if success else 'INCOMPLETE'
            plt.suptitle(f"Anchor Stability Analysis - {status}", 
                        fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout()
            
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"\n📊 Plot saved to: {save_path}")
            
            self._show_with_timeout(plt)
            
        except ImportError as ie:
            print(f"\n⚠️ matplotlib not available: {ie}")
        except Exception as e:
            print(f"\n⚠️ Error creating anchor deviation plot: {e}")
            import traceback
            traceback.print_exc()
    
    def generate_trajectory_error_plot(self, error_history: Dict,
                                       success: bool,
                                       save_path: str = "trajectory_error_analysis.png"):
        """Generate trajectory tracking error plot.
        
        Args:
            error_history: Dictionary with time_steps, errors, waypoint_idx, arm, phase
            success: Whether trajectory completed successfully
            save_path: Path to save the plot
        """
        if not error_history or len(error_history.get('time_steps', [])) == 0:
            print("\n⚠️ No trajectory error data collected - skipping trajectory plot")
            return
        
        try:
            plt = self._setup_matplotlib()
            
            print("\n📈 GENERATING TRAJECTORY ERROR GRAPH...")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            time_steps = np.array(error_history['time_steps'])
            errors = np.array(error_history['errors'])
            waypoint_idxs = np.array(error_history['waypoint_idx'])
            
            unique_waypoints = sorted(set(waypoint_idxs))
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_waypoints) + 1))
            
            # Error convergence over time
            for i, wp_idx in enumerate(unique_waypoints):
                mask = waypoint_idxs == wp_idx
                wp_steps = time_steps[mask]
                wp_errors = errors[mask]
                if len(wp_steps) > 0:
                    ax1.plot(wp_steps, wp_errors, label=f'Waypoint {wp_idx}', 
                            color=colors[i % len(colors)], alpha=0.8, linewidth=1.5)
            
            ax1.set_xlabel('Simulation Step', fontsize=12)
            ax1.set_ylabel('Position Error (m)', fontsize=12)
            ax1.set_title('Trajectory Tracking Error Over Time', fontsize=14, fontweight='bold')
            ax1.axhline(y=0.05, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Target (5cm)')
            ax1.axhline(y=0.10, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Limit (10cm)')
            ax1.legend(loc='upper right', fontsize=8)
            ax1.set_ylim(0, None)
            ax1.grid(alpha=0.3)
            
            # Initial vs final error bar chart
            initial_errors = []
            final_errors = []
            
            for wp_idx in unique_waypoints:
                mask = waypoint_idxs == wp_idx
                wp_errors = errors[mask]
                if len(wp_errors) > 0:
                    initial_errors.append(wp_errors[0])
                    final_errors.append(wp_errors[-1])
                else:
                    initial_errors.append(0)
                    final_errors.append(0)
            
            x_pos = np.arange(len(unique_waypoints))
            bar_width = 0.25
            
            bars1 = ax2.bar(x_pos - bar_width, initial_errors, bar_width, 
                           label='Initial Error (m)', color='coral', alpha=0.8)
            bars2 = ax2.bar(x_pos, final_errors, bar_width, 
                           label='Final Error (m)', color='steelblue', alpha=0.8)
            
            ax2.set_xlabel('Waypoint Index', fontsize=12)
            ax2.set_ylabel('Position Error (m)', fontsize=12)
            ax2.set_title('Initial vs Final Tracking Error per Waypoint', 
                         fontsize=14, fontweight='bold')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels([f'WP{i}' for i in unique_waypoints])
            ax2.axhline(y=0.05, color='green', linestyle='--', linewidth=1.5, label='Target (5cm)')
            ax2.axhline(y=0.15, color='orange', linestyle='--', linewidth=1.5, label='Threshold (15cm)')
            ax2.legend(loc='upper right')
            ax2.grid(axis='y', alpha=0.3)
            
            # Add value labels
            for bar, val in zip(bars1, initial_errors):
                height = bar.get_height()
                ax2.annotate(f'{val:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=7)
            for bar, val in zip(bars2, final_errors):
                height = bar.get_height()
                ax2.annotate(f'{val:.2f}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha='center', va='bottom', fontsize=7)
            
            status = 'SUCCESS' if success else 'INCOMPLETE'
            plt.suptitle(f"Trajectory Tracking Analysis - {status}", 
                        fontsize=16, fontweight='bold', y=1.02)
            plt.tight_layout()
            
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Trajectory plot saved to: {save_path}")
            
            self._show_with_timeout(plt)
            
        except ImportError as ie:
            print(f"\n⚠️ matplotlib not available: {ie}")
        except Exception as e:
            print(f"\n⚠️ Error creating trajectory plot: {e}")
            import traceback
            traceback.print_exc()
    
    def generate_dynamics_plot(self, dynamics_history: Dict,
                               success: bool,
                               save_path: str = "dynamics_analysis.png"):
        """Generate dynamics analysis plot (Anchor Forces & Body Torques).
        
        Args:
            dynamics_history: Dictionary with time_steps, anchor_forces, etc.
            success: Whether trajectory completed successfully
            save_path: Path to save the plot
        """
        if not dynamics_history or len(dynamics_history.get('time_steps', [])) == 0:
            print("\n⚠️ No dynamics data collected - skipping dynamics plot")
            return
            
        try:
            plt = self._setup_matplotlib()
            
            print("\n📈 GENERATING DYNAMICS ANALYSIS GRAPH...")
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
            
            time_steps = np.array(dynamics_history['time_steps'])
            forces = np.array(dynamics_history['anchor_forces'])
            torques = np.array(dynamics_history['body_torques'])
            arms = np.array(dynamics_history['anchor_arm'])
            
            # 1. Anchor Forces
            # Split by arm for coloring
            left_mask = arms == 'left'
            right_mask = arms == 'right'
            none_mask = arms == 'none'
            
            # Scatter plot is better for switching conditions
            if np.any(left_mask):
                ax1.scatter(time_steps[left_mask], forces[left_mask], 
                           label='Left Arm Anchor', color='#2196F3', s=10, alpha=0.6)
            if np.any(right_mask):
                ax1.scatter(time_steps[right_mask], forces[right_mask], 
                           label='Right Arm Anchor', color='#FF9800', s=10, alpha=0.6)
            
            # Also plot line for continuity
            ax1.plot(time_steps, forces, color='gray', alpha=0.3, linewidth=0.5)
            
            ax1.set_ylabel('Anchor Force (N)', fontsize=12)
            ax1.set_title('Anchor Point Reaction Forces', fontsize=14, fontweight='bold')
            ax1.legend(loc='upper right')
            ax1.grid(True, alpha=0.3)
            
            # Highlight high force events
            max_force = np.max(forces) if len(forces) > 0 else 0
            ax1.set_ylim(0, max(100.0, max_force * 1.1))
            
            # 2. Body Torque
            ax2.plot(time_steps, torques, color='purple', label='Body Orientation Torque (Z)', linewidth=1.0)
            ax2.set_xlabel('Simulation Time (s)', fontsize=12)
            ax2.set_ylabel('Torque (Nm)', fontsize=12)
            ax2.set_title('Central Body Corrective Torque (Yaw)', fontsize=14, fontweight='bold')
            ax2.axhline(y=0, color='black', alpha=0.3, linestyle='-')
            ax2.legend(loc='upper right')
            ax2.grid(True, alpha=0.3)
            
            status = 'SUCCESS' if success else 'INCOMPLETE'
            plt.suptitle(f"Robot Dynamics Analysis - {status}", 
                        fontsize=16, fontweight='bold', y=0.95)
            plt.tight_layout()
            
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"📊 Dynamics plot saved to: {save_path}")
            
            self._show_with_timeout(plt)
            
        except Exception as e:
            print(f"\n⚠️ Error creating dynamics plot: {e}")
            import traceback
            traceback.print_exc()
    
    def _show_with_timeout(self, plt):
        """Show plot with timeout then close"""
        print(f"   Displaying graph window (auto-closes in {self.plot_timeout} seconds)...")
        try:
            plt.show(block=False)
            plt.pause(self.plot_timeout)
            plt.close('all')
        except Exception as e:
            print(f"   Note: Interactive display not available ({e})")


# ============================================================================
# SCENE RENDERER - MuJoCo Visualization Geometries
# ============================================================================

class SceneRenderer:
    """Renders custom visualization geometries in MuJoCo scene."""
    
    # Colors (RGBA)
    COLORS = {
        'gray': (0.6, 0.6, 0.6, 0.4),
        'green': (0.0, 0.9, 0.0, 1.0),
        'red': (0.9, 0.0, 0.0, 1.0),
        'blue': (0.2, 0.5, 1.0, 0.9),
        'cyan': (0.0, 0.9, 0.9, 0.7),
        'yellow': (1.0, 1.0, 0.0, 0.6),
        'orange': (1.0, 0.5, 0.0, 0.8),
        'purple': (0.7, 0.0, 1.0, 0.8),
        'completed_wp': (0.3, 0.8, 0.3, 0.9),
        'upcoming_wp': (0.4, 0.4, 1.0, 0.9),
        'current_wp': (1.0, 1.0, 0.0, 1.0),
        'route_line': (0.0, 0.8, 0.8, 0.7),
        'left_workspace': (0.0, 0.5, 1.0, 0.12),
        'right_workspace': (1.0, 0.5, 0.0, 0.12),
        'bright_orange': (1.0, 0.6, 0.0, 1.0),
        'bright_blue': (0.2, 0.6, 1.0, 0.9),
    }
    
    def __init__(self, workspace_radius: float = 0.6):
        """
        Initialize scene renderer.
        
        Args:
            workspace_radius: Radius of arm workspace spheres
        """
        self.workspace_radius = workspace_radius
    
    def add_marker_geom(self, scene, pos: Tuple[float, float, float], 
                        size: float, rgba: Tuple[float, float, float, float],
                        geom_type: int = mujoco.mjtGeom.mjGEOM_SPHERE):
        """Add a marker geometry to the scene"""
        if scene.ngeom >= scene.maxgeom:
            return
        
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            geom_type,
            np.zeros(3),
            np.array(pos, dtype=np.float64),
            np.eye(3).flatten(),
            np.array(rgba, dtype=np.float32)
        )
        
        scene.geoms[scene.ngeom].size[:] = [size, size, size]
        scene.ngeom += 1
    
    def add_line_geom(self, scene, pos1: Tuple[float, float, float],
                      pos2: Tuple[float, float, float],
                      size: float, rgba: Tuple[float, float, float, float]):
        """Add a line (capsule) geometry between two points"""
        if scene.ngeom >= scene.maxgeom:
            return
        
        p1 = np.array(pos1, dtype=np.float64)
        p2 = np.array(pos2, dtype=np.float64)
        
        midpoint = (p1 + p2) / 2
        diff = p2 - p1
        length = np.linalg.norm(diff)
        
        if length < 0.001:
            return
        
        direction = diff / length
        
        z_axis = np.array([0, 0, 1])
        if np.allclose(direction, z_axis) or np.allclose(direction, -z_axis):
            rotation = np.eye(3)
            if np.dot(direction, z_axis) < 0:
                rotation[2, 2] = -1
        else:
            axis = np.cross(z_axis, direction)
            axis = axis / np.linalg.norm(axis)
            angle = np.arccos(np.clip(np.dot(z_axis, direction), -1, 1))
            
            K = np.array([[0, -axis[2], axis[1]],
                         [axis[2], 0, -axis[0]],
                         [-axis[1], axis[0], 0]])
            rotation = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K
        
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom],
            mujoco.mjtGeom.mjGEOM_CAPSULE,
            np.array([size, length / 2, 0], dtype=np.float64),
            midpoint,
            rotation.flatten(),
            np.array(rgba, dtype=np.float32)
        )
        scene.ngeom += 1
    
    def render_visualization(self, viewer, wall_positions: List,
                            start_pos: Optional[Tuple], goal_pos: Optional[Tuple],
                            current_path: Optional[List], current_wp_idx: int,
                            left_target: Optional[np.ndarray], right_target: Optional[np.ndarray],
                            left_anchor: Optional[np.ndarray], right_anchor: Optional[np.ndarray],
                            left_link4_pos: np.ndarray, right_link4_pos: np.ndarray):
        """Render all visualization geometries.
        
        Args:
            viewer: MuJoCo viewer
            wall_positions: List of (position, wall_name) tuples
            start_pos: Start position tuple
            goal_pos: Goal position tuple
            current_path: List of CrawlerState objects
            current_wp_idx: Current waypoint index
            left_target: Left arm target position
            right_target: Right arm target position
            left_anchor: Left arm anchor position
            right_anchor: Right arm anchor position
            left_link4_pos: Left arm elbow (link4) position
            right_link4_pos: Right arm elbow (link4) position
        """
        try:
            scene = viewer.user_scn
        except AttributeError:
            return
        
        scene.ngeom = 0
        
        # 1. Workspace spheres
        self.add_marker_geom(scene, tuple(left_link4_pos), 
                            size=self.workspace_radius, 
                            rgba=self.COLORS['left_workspace'])
        self.add_marker_geom(scene, tuple(right_link4_pos), 
                            size=self.workspace_radius, 
                            rgba=self.COLORS['right_workspace'])
        
        # 2. Wall grip positions
        for (pos, wall) in wall_positions:
            self.add_marker_geom(scene, pos, size=0.025, rgba=self.COLORS['gray'])
        
        # 3. Start position
        if start_pos:
            self.add_marker_geom(scene, start_pos, size=0.1, rgba=self.COLORS['green'])
        
        # 4. Goal position
        if goal_pos:
            self.add_marker_geom(scene, goal_pos, size=0.1, rgba=self.COLORS['red'])
        
        # 5. Path visualization
        if current_path and len(current_path) > 0:
            for i, wp in enumerate(current_path):
                if i < current_wp_idx:
                    color = self.COLORS['completed_wp']
                    size = 0.06
                elif i == current_wp_idx:
                    color = self.COLORS['current_wp']
                    size = 0.10
                else:
                    color = self.COLORS['upcoming_wp']
                    size = 0.07
                
                self.add_marker_geom(scene, wp.position, size=size, rgba=color)
            
            # Route lines
            for i in range(len(current_path) - 1):
                wp1 = current_path[i]
                wp2 = current_path[i + 1]
                
                if i < current_wp_idx:
                    line_color = self.COLORS['completed_wp']
                    line_width = 0.015
                else:
                    line_color = self.COLORS['route_line']
                    line_width = 0.02
                
                self.add_line_geom(scene, wp1.position, wp2.position, 
                                  size=line_width, rgba=line_color)
        
        # 6. Current arm targets
        if left_target is not None:
            self.add_marker_geom(scene, tuple(left_target), size=0.12, 
                                rgba=self.COLORS['yellow'])
        
        if right_target is not None:
            self.add_marker_geom(scene, tuple(right_target), size=0.10, 
                                rgba=self.COLORS['bright_orange'])
            self.add_marker_geom(scene, tuple(right_target), size=0.14, 
                                rgba=self.COLORS['yellow'])
        
        # 7. Trajectory line
        if left_anchor is not None and right_target is not None:
            self.add_line_geom(scene, 
                              tuple(left_anchor), 
                              tuple(right_target), 
                              size=0.02,
                              rgba=self.COLORS['bright_blue'])


# ============================================================================
# FLOW DIAGRAM GENERATOR
# ============================================================================

def generate_flow_diagram(output_path: str = "wall_crawler_architecture.png"):
    """Generate a flow diagram of the wall crawler code architecture.
    
    Args:
        output_path: Path to save the diagram image
    """
    try:
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
        
        fig, ax = plt.subplots(figsize=(16, 12))
        ax.set_xlim(0, 16)
        ax.set_ylim(0, 12)
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Colors
        colors = {
            'main': '#4CAF50',      # Green - main simulation
            'controller': '#2196F3', # Blue - arm controller
            'viz': '#FF9800',        # Orange - visualization
            'state': '#9C27B0',      # Purple - state machine
            'data': '#607D8B',       # Gray - data structures
        }
        
        def add_box(x, y, w, h, text, color, fontsize=10):
            box = FancyBboxPatch((x, y), w, h, 
                                boxstyle="round,pad=0.05,rounding_size=0.2",
                                facecolor=color, edgecolor='black', linewidth=2,
                                alpha=0.8)
            ax.add_patch(box)
            ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                   fontsize=fontsize, fontweight='bold', wrap=True)
        
        def add_arrow(start, end, color='black'):
            arrow = FancyArrowPatch(start, end,
                                   arrowstyle='-|>',
                                   mutation_scale=15,
                                   lw=2, color=color)
            ax.add_patch(arrow)
        
        # Title
        ax.text(8, 11.5, 'Wall Crawler Robot - Code Architecture', 
               ha='center', va='center', fontsize=18, fontweight='bold')
        
        # Main modules
        add_box(6, 9, 4, 1.5, 'wall_crawler_mujoco.py\n(Main Simulation)', colors['main'])
        
        add_box(1, 6, 4, 2, 'arm_controller.py\n\n• IK Computation\n• Gripper Control\n• Anchor Management', colors['controller'])
        
        add_box(11, 6, 4, 2, 'visualization.py\n\n• SimulationLogger\n• PlotGenerator\n• SceneRenderer', colors['viz'])
        
        # State machine
        add_box(6, 5, 4, 2.5, 'State Machine\n\n• reaching_anchor\n• reaching_waypoint\n• trajectory_complete\n• failed', colors['state'])
        
        # Data flow
        add_box(1, 2.5, 3, 1.5, 'MuJoCo\nModel/Data', colors['data'])
        add_box(6, 2.5, 4, 1.5, 'Path Planner\n(A* Algorithm)', colors['data'])
        add_box(12, 2.5, 3, 1.5, 'Run History\n(JSON)', colors['data'])
        
        # Helper classes
        add_box(1, 0.5, 3, 1.5, 'CrawlerState\nWallPositionMapper', colors['data'])
        add_box(12, 0.5, 3, 1.5, 'ArmControllerConfig', colors['data'])
        
        # Arrows
        # Main to controller
        add_arrow((6, 9.75), (5, 8), colors['controller'])
        ax.text(4.5, 9, 'imports', fontsize=8, color=colors['controller'])
        
        # Main to visualization
        add_arrow((10, 9.75), (11, 8), colors['viz'])
        ax.text(11, 9, 'imports', fontsize=8, color=colors['viz'])
        
        # Main to state machine
        add_arrow((8, 9), (8, 7.5), colors['state'])
        
        # Controller to MuJoCo
        add_arrow((3, 6), (3, 4), colors['controller'])
        
        # State to path planner
        add_arrow((8, 5), (8, 4), colors['state'])
        
        # Visualization to history
        add_arrow((13.5, 6), (13.5, 4), colors['viz'])
        
        # Controller to config
        add_arrow((3, 6), (13.5, 2), colors['controller'])
        
        # Legend
        legend_y = 0.3
        for i, (name, color) in enumerate(colors.items()):
            rect = plt.Rectangle((0.5 + i*3, legend_y), 0.5, 0.3, color=color, alpha=0.8)
            ax.add_patch(rect)
            labels = {'main': 'Main', 'controller': 'Controller', 'viz': 'Visualization',
                     'state': 'State Machine', 'data': 'Data/Utils'}
            ax.text(1.1 + i*3, legend_y + 0.15, labels[name], fontsize=8, va='center')
        
        plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close()
        
        print(f"📊 Flow diagram saved to: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"⚠️ Error generating flow diagram: {e}")
        import traceback
        traceback.print_exc()
        return None
