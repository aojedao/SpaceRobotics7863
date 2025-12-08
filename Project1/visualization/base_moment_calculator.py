"""
Base Reaction Moment Calculator
================================
Computes reaction moments at robot base from end-effector forces.
Includes proper coordinate transformation from EE frame to base frame.
"""

import mujoco
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple

class BaseReactionMomentCalculator:
    """Calculate base reaction moments from end-effector forces."""
    
    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData):
        self.model = model
        self.data = data
        
        # Find relevant body/site IDs
        self.left_ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_gripper_center")
        self.right_ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_gripper_center")
        
        # Logging
        self.time_log = []
        self.left_moment_log = []
        self.right_moment_log = []
        self.total_moment_log = []
        
    def get_ee_to_base_rotation(self, ee_site_id: int) -> np.ndarray:
        """
        Get rotation matrix from end-effector frame to base frame.
        
        Returns:
            3x3 rotation matrix R_base_ee
        """
        # Get EE rotation matrix (3x3)
        ee_rotmat = self.data.site_xmat[ee_site_id].reshape(3, 3)
        
        # Get base rotation matrix from quaternion
        base_quat = self.data.qpos[3:7]  # [w, x, y, z] format
        base_rotmat = np.zeros(9)
        mujoco.mju_quat2Mat(base_rotmat, base_quat)
        base_rotmat = base_rotmat.reshape(3, 3)
        
        # Transform: R_base_ee = R_base_world @ R_world_ee
        # Since base is our reference, we want: R_base @ R_ee^T
        R_base_ee = base_rotmat @ ee_rotmat.T
        
        return R_base_ee
    
    def compute_reaction_moment(self, 
                                ee_site_id: int, 
                                ee_force: np.ndarray,
                                ee_torque: np.ndarray = None) -> np.ndarray:
        """
        Compute reaction moment at base from end-effector force.
        
        Args:
            ee_site_id: MuJoCo site ID for end-effector
            ee_force: Force at EE in EE frame [Fx, Fy, Fz] (N)
            ee_torque: Optional torque at EE in EE frame [Tx, Ty, Tz] (N·m)
            
        Returns:
            Moment at base in base frame [Mx, My, Mz] (N·m)
        """
        # Get EE position in world frame
        ee_pos_world = self.data.site_xpos[ee_site_id].copy()
        
        # Get base position in world frame
        base_pos_world = self.data.qpos[0:3].copy()
        
        # Position vector from base to EE (in world frame)
        r_base_ee_world = ee_pos_world - base_pos_world
        
        # Get rotation matrix from EE frame to base frame
        R_base_ee = self.get_ee_to_base_rotation(ee_site_id)
        
        # Transform force from EE frame to base frame
        # F_base = R_base_ee @ F_ee
        ee_force_base = R_base_ee @ ee_force
        
        # Compute moment: M = r × F (cross product)
        # r is in world frame, F is in base frame
        # Need to transform r to base frame as well
        base_rotmat = np.zeros(9)
        mujoco.mju_quat2Mat(base_rotmat, self.data.qpos[3:7])
        base_rotmat = base_rotmat.reshape(3, 3)
        r_base_ee_base = base_rotmat @ r_base_ee_world
        
        # Moment from force
        moment_from_force = np.cross(r_base_ee_base, ee_force_base)
        
        # Add direct torque contribution if provided
        if ee_torque is not None:
            ee_torque_base = R_base_ee @ ee_torque
            total_moment = moment_from_force + ee_torque_base
        else:
            total_moment = moment_from_force
        
        return total_moment
    
    def log_current_state(self, 
                          left_force: np.ndarray, 
                          right_force: np.ndarray,
                          left_torque: np.ndarray = None,
                          right_torque: np.ndarray = None):
        """Log reaction moments at current timestep."""
        
        # Compute moments from each arm
        left_moment = self.compute_reaction_moment(self.left_ee_site_id, left_force, left_torque)
        right_moment = self.compute_reaction_moment(self.right_ee_site_id, right_force, right_torque)
        
        # Total moment at base
        total_moment = left_moment + right_moment
        
        # Log
        self.time_log.append(self.data.time)
        self.left_moment_log.append(left_moment.copy())
        self.right_moment_log.append(right_moment.copy())
        self.total_moment_log.append(total_moment.copy())
        
        return total_moment
    
    def plot_results(self, save_path: str = None):
        """Plot reaction moments over time."""
        
        if len(self.time_log) == 0:
            print("No data to plot")
            return
        
        # Convert to numpy arrays
        times = np.array(self.time_log)
        left_moments = np.array(self.left_moment_log)
        right_moments = np.array(self.right_moment_log)
        total_moments = np.array(self.total_moment_log)
        
        # Compute magnitudes
        left_mag = np.linalg.norm(left_moments, axis=1)
        right_mag = np.linalg.norm(right_moments, axis=1)
        total_mag = np.linalg.norm(total_moments, axis=1)
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Plot 1: Total moment magnitude
        ax1 = axes[0, 0]
        ax1.plot(times, total_mag, 'b-', linewidth=2, label='Total Magnitude')
        ax1.set_xlabel('Time (s)')
        ax1.set_ylabel('Moment Magnitude (N·m)')
        ax1.set_title('Total Base Reaction Moment')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Individual arm contributions
        ax2 = axes[0, 1]
        ax2.plot(times, left_mag, 'r-', linewidth=1.5, label='Left Arm')
        ax2.plot(times, right_mag, 'g-', linewidth=1.5, label='Right Arm')
        ax2.plot(times, total_mag, 'b--', linewidth=2, label='Total')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Moment Magnitude (N·m)')
        ax2.set_title('Arm Contributions to Base Moment')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Moment components (X, Y, Z)
        ax3 = axes[1, 0]
        ax3.plot(times, total_moments[:, 0], 'r-', label='Mx', linewidth=1.5)
        ax3.plot(times, total_moments[:, 1], 'g-', label='My', linewidth=1.5)
        ax3.plot(times, total_moments[:, 2], 'b-', label='Mz', linewidth=1.5)
        ax3.set_xlabel('Time (s)')
        ax3.set_ylabel('Moment Component (N·m)')
        ax3.set_title('Base Moment Components (Base Frame)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='k', linestyle='--', alpha=0.3)
        
        # Plot 4: Compensation effectiveness (compare with/without)
        ax4 = axes[1, 1]
        # For this plot, we'd need data from both compensated and uncompensated runs
        # For now, show the magnitude over time with threshold
        ax4.plot(times, total_mag, 'b-', linewidth=2)
        threshold = 0.5  # Example threshold (N·m)
        ax4.axhline(y=threshold, color='r', linestyle='--', linewidth=2, label=f'Threshold ({threshold} N·m)')
        ax4.fill_between(times, 0, threshold, alpha=0.2, color='green', label='Acceptable Range')
        ax4.set_xlabel('Time (s)')
        ax4.set_ylabel('Moment Magnitude (N·m)')
        ax4.set_title('Moment Stability (with Compensation)')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Plot saved to: {save_path}")
        
        plt.show()
        
    def print_summary(self):
        """Print statistical summary of reaction moments."""
        
        if len(self.time_log) == 0:
            print("No data available")
            return
        
        total_moments = np.array(self.total_moment_log)
        total_mag = np.linalg.norm(total_moments, axis=1)
        
        print("\n" + "="*60)
        print("BASE REACTION MOMENT SUMMARY")
        print("="*60)
        print(f"Duration: {self.time_log[-1]:.2f} seconds")
        print(f"Samples: {len(self.time_log)}")
        print(f"\nMoment Magnitude Statistics:")
        print(f"  Mean: {np.mean(total_mag):.4f} N·m")
        print(f"  Std:  {np.std(total_mag):.4f} N·m")
        print(f"  Max:  {np.max(total_mag):.4f} N·m")
        print(f"  Min:  {np.min(total_mag):.4f} N·m")
        print(f"\nMoment Components (Mean):")
        print(f"  Mx: {np.mean(total_moments[:, 0]):.4f} N·m")
        print(f"  My: {np.mean(total_moments[:, 1]):.4f} N·m")
        print(f"  Mz: {np.mean(total_moments[:, 2]):.4f} N·m")
        print("="*60)


# Example usage function
def example_usage():
    """Example of how to use the BaseReactionMomentCalculator."""
    
    print("This is a utility module.")
    print("Import it in your simulation script like:")
    print()
    print("  from base_moment_calculator import BaseReactionMomentCalculator")
    print()
    print("  calculator = BaseReactionMomentCalculator(model, data)")
    print()
    print("  # In your simulation loop:")
    print("  ee_force = np.array([Fx, Fy, Fz])  # Force at gripper")
    print("  ee_torque = np.array([Tx, Ty, Tz])  # Torque at gripper (optional)")
    print("  calculator.log_current_state(left_force, right_force, left_torque, right_torque)")
    print()
    print("  # After simulation:")
    print("  calculator.print_summary()")
    print("  calculator.plot_results('base_moments.png')")
    print()


if __name__ == "__main__":
    example_usage()
