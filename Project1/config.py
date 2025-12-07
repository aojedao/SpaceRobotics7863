
import numpy as np

# ============================================================================
# ROBOT CONTROL GAINS
# ============================================================================

# Cartesian Position Control
KP_POSITION = 800.0       # Proportional gain for position
KD_POSITION = 50.0        # Damping gain for position
KP_ORIENTATION = 50.0     # Proportional gain for orientation
KD_ORIENTATION = 25.0     # Damping gain for orientation
LAMBDA_DLS = 0.008        # Damping factor for Damped Least Squares IK

# Per-joint gain scaling (Joints 1-2 need more authority as they move the whole arm)
# [joint1, joint2, joint3, joint4, joint5, joint6, joint7]
JOINT_GAIN_SCALE = np.array([3.0, 2.5, 1.5, 1.2, 1.0, 0.8, 0.6])

# Anchoring Control
KP_ANCHOR = 1500.0        # Stiffness for maintaining anchor
KD_ANCHOR = 50.0          # Damping for anchor stability

# Body Control (Locomotion)
KP_BODY_MOVE = 400.0      # Proportional gain for moving body
KD_BODY_MOVE = 20.0       # Damping for body movement

# Body Orientation (Keep upright/aimed)
KP_BODY_ORIENT = 80.0
KD_BODY_ORIENT = 18.0

# ============================================================================
# DIAGNOSTICS CONFIGURATION
# ============================================================================
PRINT_CROSSING_DIAGNOSTICS = True
