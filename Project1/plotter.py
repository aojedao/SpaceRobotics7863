
import numpy as np
import mujoco

# ============================================================================
# PLOTTING AND VISUALIZATION UTILITIES
# ============================================================================

def add_marker_geom(scene, pos, size, rgba, geom_type=mujoco.mjtGeom.mjGEOM_SPHERE):
    """Add a marker geometry to the scene for visualization"""
    if scene.ngeom >= scene.maxgeom:
        return  # Scene is full
    
    # Initialize the geometry
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        geom_type,
        np.zeros(3),  # size (will be set below)
        np.array(pos, dtype=np.float64),  # position
        np.eye(3).flatten(),  # rotation matrix (identity)
        np.array(rgba, dtype=np.float32)  # color
    )
    
    # Set size for sphere
    scene.geoms[scene.ngeom].size[:] = [size, size, size]
    scene.ngeom += 1

def add_line_geom(scene, pos1, pos2, size, rgba):
    """Add a line (capsule) geometry between two points"""
    if scene.ngeom >= scene.maxgeom:
        return
    
    p1 = np.array(pos1, dtype=np.float64)
    p2 = np.array(pos2, dtype=np.float64)
    
    # MuJoCo's capsule geom works by defining pos and rotation, and length
    # But mjv_initGeom for CAPSULE expects pos to be center.
    # We can use mjGEOM_CAPSULE and compute frame, or use mjv_connector for lines.
    
    # Better to use mjv_makeConnector for lines/arrows in visualization if available,
    # but manually constructing a cylinder/capsule is robust.
    
    # Simplified: Initialize as a capsule connector (using start/end points)
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3),
        np.zeros(3),
        np.zeros(9),
        np.array(rgba, dtype=np.float32)
    )
    mujoco.mjv_connector(
        scene.geoms[scene.ngeom],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        size,
        p1,
        p2
    )
    scene.ngeom += 1

def create_anchor_deviation_plot(anchor_deviation_history, trajectory_success, show_interactive=True):
    """Create matplotlib graph of max deviations"""
    try:
        import matplotlib
        try:
            matplotlib.use('TkAgg')
        except:
            pass  # Use default backend if TkAgg not available
        import matplotlib.pyplot as plt
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        max_deviations = []
        wp_labels = []
        
        for wp_idx in sorted(anchor_deviation_history.keys()):
            data = anchor_deviation_history[wp_idx]
            arm = data['arm']
            max_dev = data['max_deviation']
            
            max_deviations.append(max_dev)
            wp_labels.append(f"WP{wp_idx+1}\n({arm[0].upper()})")
        
        # Bar chart of max deviations
        colors = ['#2196F3' if 'L' in label else '#FF9800' for label in wp_labels]
        bars = ax1.bar(range(len(max_deviations)), max_deviations, color=colors, edgecolor='black', linewidth=1.2)
        ax1.set_xticks(range(len(max_deviations)))
        ax1.set_xticklabels(wp_labels)
        ax1.set_xlabel('Waypoint (Arm)', fontsize=12)
        ax1.set_ylabel('Max End-Effector Deviation (m)', fontsize=12)
        ax1.set_title('Maximum EE Deviation from Anchor Point per Waypoint', fontsize=14, fontweight='bold')
        ax1.axhline(y=0.05, color='green', linestyle='--', linewidth=1.5, label='Good (<5cm)')
        ax1.axhline(y=0.10, color='orange', linestyle='--', linewidth=1.5, label='Acceptable (<10cm)')
        ax1.axhline(y=0.20, color='red', linestyle='--', linewidth=1.5, label='Limit (20cm)')
        ax1.legend(loc='upper right')
        ax1.set_ylim(0, max(0.25, max(max_deviations) * 1.2) if max_deviations else 0.25)
        ax1.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, max_deviations):
            height = bar.get_height()
            ax1.annotate(f'{val:.3f}m',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=9)
        
        # Time series of deviations
        all_deviations = []
        all_wp_indices = []
        for wp_idx in sorted(anchor_deviation_history.keys()):
            devs = anchor_deviation_history[wp_idx]['deviations']
            all_deviations.extend(devs)
            all_wp_indices.extend([wp_idx] * len(devs))
        
        if all_deviations:
            # Color-code by waypoint
            ax2.scatter(range(len(all_deviations)), all_deviations, c=all_wp_indices, cmap='viridis', alpha=0.5, s=2)
            ax2.set_xlabel('Time Step (during anchoring)', fontsize=12)
            ax2.set_ylabel('EE Deviation from Anchor (m)', fontsize=12)
            ax2.set_title('End-Effector Deviation Over Time (All Anchors)', fontsize=14, fontweight='bold')
            ax2.axhline(y=0.05, color='green', linestyle='--', linewidth=1.5, alpha=0.7)
            ax2.axhline(y=0.10, color='orange', linestyle='--', linewidth=1.5, alpha=0.7)
            ax2.set_ylim(0, max(0.15, max(all_deviations) * 1.2) if all_deviations else 0.15)
            ax2.grid(alpha=0.3)
            
            # Add colorbar
            sm = plt.cm.ScalarMappable(cmap='viridis', norm=plt.Normalize(vmin=min(all_wp_indices), vmax=max(all_wp_indices)))
            sm.set_array([])
            cbar = plt.colorbar(sm, ax=ax2)
            cbar.set_label('Waypoint Index')
        
        plt.suptitle(f"Anchor Stability Analysis - {'SUCCESS' if trajectory_success else 'INCOMPLETE'}", 
                   fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save plot to file first (always works)
        plot_filename = "anchor_deviation_analysis.png"
        plt.savefig(plot_filename, dpi=150, bbox_inches='tight')
        print(f"\n📊 Plot saved to: {plot_filename}")
        
        # Try to display interactively with timeout
        if show_interactive:
            print("   Displaying graph window (auto-closes in 5 seconds)...")
            try:
                plt.show(block=False)
                plt.pause(5)  # Show for 5 seconds then continue
                plt.close('all')
            except Exception as show_err:
                print(f"   Note: Interactive display not available ({show_err})")
                print(f"   View the saved file: {plot_filename}")
    
    except ImportError as ie:
        print(f"\n⚠️ matplotlib not available for anchor deviation plotting: {ie}")
    except Exception as e:
        print(f"\n⚠️ Error creating anchor deviation plot: {e}")
        import traceback
        traceback.print_exc()

def create_trajectory_error_plot(trajectory_error_history, trajectory_success, show_interactive=True):
    if not trajectory_error_history or len(trajectory_error_history['time_steps']) == 0:
        print("\n⚠️ No trajectory error data collected - skipping trajectory plot")
        return

    try:
        import matplotlib
        try:
            matplotlib.use('TkAgg')
        except:
            pass
        import matplotlib.pyplot as plt
        
        print("\n📈 GENERATING TRAJECTORY ERROR GRAPH...")
        
        fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Extract data
        time_steps = np.array(trajectory_error_history['time_steps'])
        errors = np.array(trajectory_error_history['errors'])
        waypoint_idxs = np.array(trajectory_error_history['waypoint_idx'])
        
        # Get unique waypoints
        unique_waypoints = sorted(set(waypoint_idxs))
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_waypoints) + 1))
        
        # Left plot: Trajectory error convergence over time for each waypoint
        for i, wp_idx in enumerate(unique_waypoints):
            mask = waypoint_idxs == wp_idx
            wp_steps = time_steps[mask]
            wp_errors = errors[mask]
            if len(wp_steps) > 0:
                ax3.plot(wp_steps, wp_errors, label=f'Waypoint {wp_idx}', 
                        color=colors[i % len(colors)], alpha=0.8, linewidth=1.5)
        
        ax3.set_xlabel('Simulation Step', fontsize=12)
        ax3.set_ylabel('Position Error (m)', fontsize=12)
        ax3.set_title('Trajectory Tracking Error Over Time', fontsize=14, fontweight='bold')
        ax3.axhline(y=0.05, color='green', linestyle='--', linewidth=1.5, alpha=0.7, label='Target (5cm)')
        ax3.axhline(y=0.10, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Limit (10cm)')
        ax3.legend(loc='upper right', fontsize=8)
        ax3.set_ylim(0, None)
        ax3.grid(alpha=0.3)
        
        # Right plot: Initial error, final error per waypoint
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
        
        bars1 = ax4.bar(x_pos - bar_width, initial_errors, bar_width, 
                       label='Initial Error (m)', color='coral', alpha=0.8)
        bars2 = ax4.bar(x_pos, final_errors, bar_width, 
                       label='Final Error (m)', color='steelblue', alpha=0.8)
        
        ax4.set_xlabel('Waypoint Index', fontsize=12)
        ax4.set_ylabel('Position Error (m)', fontsize=12)
        ax4.set_title('Initial vs Final Tracking Error per Waypoint', fontsize=14, fontweight='bold')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels([f'WP{i}' for i in unique_waypoints])
        ax4.axhline(y=0.05, color='green', linestyle='--', linewidth=1.5, label='Target (5cm)')
        ax4.axhline(y=0.15, color='orange', linestyle='--', linewidth=1.5, label='Threshold (15cm)')
        ax4.legend(loc='upper right')
        ax4.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, val in zip(bars1, initial_errors):
            height = bar.get_height()
            ax4.annotate(f'{val:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=7)
        for bar, val in zip(bars2, final_errors):
            height = bar.get_height()
            ax4.annotate(f'{val:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3),
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=7)
        
        plt.suptitle(f"Trajectory Tracking Analysis - {'SUCCESS' if trajectory_success else 'INCOMPLETE'}", 
                   fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()
        
        # Save trajectory plot
        traj_plot_filename = "trajectory_error_analysis.png"
        plt.savefig(traj_plot_filename, dpi=150, bbox_inches='tight')
        print(f"📊 Trajectory plot saved to: {traj_plot_filename}")
        
        if show_interactive:
            try:
                plt.show(block=False)
                plt.pause(5)  # Show for 5 seconds then continue
                plt.close('all')
            except Exception as show_err:
                print(f"   Note: Interactive display not available ({show_err})")
                
    except ImportError as ie:
        print(f"\n⚠️ matplotlib not available for trajectory plotting: {ie}")
    except Exception as e:
        print(f"\n⚠️ Error creating trajectory plot: {e}")
        import traceback
        traceback.print_exc()

