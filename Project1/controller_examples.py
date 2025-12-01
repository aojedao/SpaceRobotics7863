#!/usr/bin/env python3
"""
Examples of using the controller module

This file demonstrates various ways to use the new controller framework.
"""

import numpy as np
import mujoco
from controller import ControllerFactory, PositionController, TorqueBalancingController


# ============================================================================
# Example 1: Basic Usage with Integrated Simulation
# ============================================================================

def example_1_basic_usage():
    """
    Most basic usage - run simulation with default position controller
    """
    from integrated_simulation import IntegratedZeroGravitySimulation
    
    # Create simulation (uses position controller by default)
    sim = IntegratedZeroGravitySimulation()
    
    # Run for 10 seconds
    sim.run_simulation(duration=10)
    

def example_1_torque_balancing():
    """
    Basic usage with torque balancing controller
    """
    from integrated_simulation import IntegratedZeroGravitySimulation
    
    # Create simulation with torque balancing
    sim = IntegratedZeroGravitySimulation(controller_type='torque_balancing')
    
    # Run indefinitely (Ctrl+C to stop)
    sim.run_simulation()


# ============================================================================
# Example 2: Custom Controller Configuration
# ============================================================================

def example_2_custom_configuration():
    """
    Advanced usage with custom controller parameters
    """
    from integrated_simulation import IntegratedZeroGravitySimulation
    import mujoco.viewer
    
    class CustomSimulation(IntegratedZeroGravitySimulation):
        def __init__(self):
            super().__init__(controller_type='torque_balancing')
            
            # Fine-tune controller parameters AFTER initialization
            self.controller.kp_position = 120.0      # Higher gain = stiffer
            self.controller.kd_position = 30.0       # Higher damping
            self.controller.max_joint_velocity = 3.0
            self.controller.torque_balance_gain = 0.75  # Stronger balancing
    
    sim = CustomSimulation()
    sim.run_simulation(duration=15)


# ============================================================================
# Example 3: Direct Controller Usage (without simulation)
# ============================================================================

def example_3_standalone_controller():
    """
    Use controllers directly without the full simulation framework
    """
    
    # Load model
    model = mujoco.MjModel.from_xml_path("integrated_model.xml")
    data = mujoco.MjData(model)
    
    # Create position controller
    controller = ControllerFactory.create_controller(
        'position',
        model,
        data,
        kp_position=100.0,
        kd_position=20.0
    )
    
    # Simulation loop
    for step in range(10000):
        # Compute control
        control_data = controller.compute_control()
        
        # Step simulation
        mujoco.mj_step(model, data)
        
        # Can access control data for analysis
        if control_data and step % 100 == 0:
            pos_error = np.linalg.norm(control_data['position_error'])
            print(f"Step {step}: Position Error = {pos_error:.4f} m")


def example_3_torque_controller_standalone():
    """
    Standalone usage of torque balancing controller
    """
    
    model = mujoco.MjModel.from_xml_path("integrated_model.xml")
    data = mujoco.MjData(model)
    
    # Create torque balancing controller
    controller = ControllerFactory.create_controller(
        'torque_balancing',
        model,
        data,
        kp_position=100.0,
        kd_position=20.0,
        torque_balance_gain=0.5,
        shear_force_threshold=0.1
    )
    
    # Track metrics
    shear_forces = []
    position_errors = []
    
    # Simulation loop
    for step in range(10000):
        control_data = controller.compute_control()
        mujoco.mj_step(model, data)
        
        if control_data:
            # Collect torque-specific data
            shear_forces.append(control_data.get('shear_force', 0))
            position_errors.append(np.linalg.norm(control_data['position_error']))
            
            if step % 1000 == 0 and step > 0:
                avg_shear = np.mean(shear_forces[-1000:])
                avg_error = np.mean(position_errors[-1000:])
                print(f"Step {step}:")
                print(f"  Avg Shear Force: {avg_shear:.6f}")
                print(f"  Avg Position Error: {avg_error:.4f} m")


# ============================================================================
# Example 4: Comparing Controllers
# ============================================================================

def example_4_controller_comparison():
    """
    Run side-by-side comparison of controllers
    """
    import matplotlib.pyplot as plt
    
    results = {}
    
    # Test each controller
    for controller_type in ['position', 'torque_balancing']:
        print(f"\n{'='*60}")
        print(f"Testing {controller_type.upper()} controller...")
        print(f"{'='*60}")
        
        model = mujoco.MjModel.from_xml_path("integrated_model.xml")
        data = mujoco.MjData(model)
        
        controller = ControllerFactory.create_controller(
            controller_type,
            model,
            data,
            kp_position=100.0,
            kd_position=20.0
        )
        
        # Collect data
        time_steps = []
        pos_errors = []
        orient_errors = []
        joint_efforts = []
        shear_forces = []
        
        # Run simulation
        for step in range(5000):
            control_data = controller.compute_control()
            mujoco.mj_step(model, data)
            
            if control_data:
                time_steps.append(step)
                pos_errors.append(np.linalg.norm(control_data['position_error']))
                orient_errors.append(np.linalg.norm(control_data['orientation_error']))
                joint_efforts.append(np.linalg.norm(data.ctrl[:7]))
                
                if 'shear_force' in control_data:
                    shear_forces.append(control_data['shear_force'])
        
        results[controller_type] = {
            'time': time_steps,
            'pos_error': pos_errors,
            'orient_error': orient_errors,
            'effort': joint_efforts,
            'shear': shear_forces if shear_forces else None
        }
        
        print(f"\nResults for {controller_type}:")
        print(f"  Final position error: {pos_errors[-1]:.4f} m")
        print(f"  Final orientation error: {orient_errors[-1]:.4f} rad")
        print(f"  Mean joint effort: {np.mean(joint_efforts):.4f}")
        if shear_forces:
            print(f"  Mean shear force: {np.mean(shear_forces):.6f}")
    
    # Plot comparison
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Position error
    for ctrl_type in results:
        axes[0, 0].plot(results[ctrl_type]['time'], results[ctrl_type]['pos_error'], 
                       label=ctrl_type, alpha=0.7)
    axes[0, 0].set_ylabel('Position Error (m)')
    axes[0, 0].set_title('Position Tracking Error')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Orientation error
    for ctrl_type in results:
        axes[0, 1].plot(results[ctrl_type]['time'], results[ctrl_type]['orient_error'],
                       label=ctrl_type, alpha=0.7)
    axes[0, 1].set_ylabel('Orientation Error (rad)')
    axes[0, 1].set_title('Orientation Tracking Error')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Joint effort
    for ctrl_type in results:
        axes[1, 0].plot(results[ctrl_type]['time'], results[ctrl_type]['effort'],
                       label=ctrl_type, alpha=0.7)
    axes[1, 0].set_ylabel('Joint Effort Norm')
    axes[1, 0].set_title('Control Effort')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Shear force (if available)
    if results['torque_balancing']['shear'] is not None:
        axes[1, 1].plot(results['torque_balancing']['time'][:len(results['torque_balancing']['shear'])],
                       results['torque_balancing']['shear'], label='torque_balancing', alpha=0.7)
        axes[1, 1].set_ylabel('Base Shear Force')
        axes[1, 1].set_title('Base Shear Force Evolution')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig('controller_comparison.png', dpi=150)
    plt.show()


# ============================================================================
# Example 5: Creating a Custom Controller
# ============================================================================

def example_5_custom_controller():
    """
    Example of creating and using a completely custom controller
    """
    from controller import BaseController, ControllerFactory
    
    class MyAdmittanceController(BaseController):
        """
        Simple admittance controller that responds to estimated forces
        """
        
        def __init__(self, model, data, **kwargs):
            super().__init__(model, data)
            self.stiffness = kwargs.get('stiffness', 50.0)
            self.damping = kwargs.get('damping', 10.0)
            self.target_position = None
        
        def compute_control(self):
            if not self.enabled:
                return
            
            # Get current state
            current_pos = self.get_end_effector_position()
            current_orient = self.get_end_effector_orientation()
            self.target_position = self.get_target_position()
            target_orient = self.get_target_orientation()
            
            # Simple position error
            pos_error = self.target_position - current_pos
            
            # Admittance response (less stiff than position control)
            desired_vel = pos_error * self.stiffness / 100.0
            
            # Simple joint space control (for demo, just using pseudo-inverse)
            J = self.compute_jacobian()
            try:
                J_inv = np.linalg.pinv(J)
                joint_vels = J_inv @ np.concatenate([desired_vel, np.zeros(3)])
            except:
                joint_vels = np.zeros(7)
            
            # Apply control
            joint_vels = np.clip(joint_vels, -2.0, 2.0)
            self.data.ctrl[:7] = joint_vels
            
            return {
                'current_pos': current_pos,
                'current_orient': current_orient,
                'target_orient': target_orient,
                'position_error': pos_error,
                'orientation_error': np.zeros(3),
                'current_joint_vel': self.data.qvel[:7],
                'current_angular_vel': np.zeros(3)
            }
    
    # Register custom controller
    ControllerFactory.AVAILABLE_CONTROLLERS['admittance'] = MyAdmittanceController
    
    # Use it
    model = mujoco.MjModel.from_xml_path("integrated_model.xml")
    data = mujoco.MjData(model)
    
    controller = ControllerFactory.create_controller(
        'admittance',
        model,
        data,
        stiffness=60.0,
        damping=15.0
    )
    
    # Run simulation
    for step in range(1000):
        controller.compute_control()
        mujoco.mj_step(model, data)


# ============================================================================
# Example 6: Data Analysis and Visualization
# ============================================================================

def example_6_data_analysis():
    """
    Collect data from a controller run and perform analysis
    """
    import matplotlib.pyplot as plt
    from integrated_simulation import IntegratedZeroGravitySimulation
    
    class DataCollectingSimulation(IntegratedZeroGravitySimulation):
        def __init__(self):
            super().__init__(controller_type='torque_balancing')
            self.performance_metrics = {
                'success': False,
                'final_pos_error': None,
                'max_pos_error': None,
                'mean_pos_error': None,
                'mean_shear_force': None,
                'convergence_time': None
            }
        
        def calculate_metrics(self):
            """Calculate performance metrics from collected data"""
            if len(self.position_error_data) < 10:
                return
            
            pos_errors = np.array([np.linalg.norm(e) for e in self.position_error_data])
            
            self.performance_metrics['final_pos_error'] = pos_errors[-1]
            self.performance_metrics['max_pos_error'] = np.max(pos_errors)
            self.performance_metrics['mean_pos_error'] = np.mean(pos_errors)
            
            # Convergence time (when error drops below 0.1m)
            converged = np.where(pos_errors < 0.1)[0]
            if len(converged) > 0:
                self.performance_metrics['convergence_time'] = converged[0] * 0.002  # timestep
                self.performance_metrics['success'] = True
            
            if len(self.shear_force_data) > 0:
                self.performance_metrics['mean_shear_force'] = np.mean(self.shear_force_data)
            
            print("\n" + "="*60)
            print("PERFORMANCE METRICS")
            print("="*60)
            for key, value in self.performance_metrics.items():
                if value is not None:
                    if isinstance(value, float):
                        print(f"{key:.<40} {value:.6f}")
                    else:
                        print(f"{key:.<40} {value}")
    
    sim = DataCollectingSimulation()
    sim.run_simulation(duration=20)
    sim.calculate_metrics()


# ============================================================================
# Command-line Examples
# ============================================================================

"""
Command-line usage examples:

# Run with default position controller
python integrated_simulation.py

# Run with torque balancing controller
python integrated_simulation.py --controller torque_balancing

# Run for specific duration
python integrated_simulation.py --controller position --duration 30

# Get help
python integrated_simulation.py --help

# Run with both controllers for 10 seconds each
for ctrl in position torque_balancing; do
    echo "Running $ctrl..."
    python integrated_simulation.py --controller $ctrl --duration 10
done
"""


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        example_num = sys.argv[1]
        if example_num == "1":
            example_1_basic_usage()
        elif example_num == "1b":
            example_1_torque_balancing()
        elif example_num == "2":
            example_2_custom_configuration()
        elif example_num == "3":
            example_3_standalone_controller()
        elif example_num == "3b":
            example_3_torque_controller_standalone()
        elif example_num == "4":
            example_4_controller_comparison()
        elif example_num == "5":
            example_5_custom_controller()
        elif example_num == "6":
            example_6_data_analysis()
        else:
            print("Unknown example number")
    else:
        print(__doc__)
        print("\nUsage: python controller_examples.py <example_number>")
        print("\nAvailable examples:")
        print("  1   - Basic usage with position controller")
        print("  1b  - Basic usage with torque balancing")
        print("  2   - Custom controller configuration")
        print("  3   - Standalone controller usage")
        print("  3b  - Standalone torque balancing")
        print("  4   - Controller comparison")
        print("  5   - Creating custom controller")
        print("  6   - Data analysis and visualization")
