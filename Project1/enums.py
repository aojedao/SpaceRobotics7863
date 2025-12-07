
from enum import Enum, auto

class RobotState(Enum):
    """State machine states for the wall-crawling robot"""
    IDLE = auto()                    # Robot is stationary, waiting for commands
    PLANNING = auto()                # Computing path to goal
    READY_TO_MOVE = auto()           # Path computed, ready to execute
    MOVING_LEFT_ARM = auto()         # Left arm is moving to new position
    MOVING_RIGHT_ARM = auto()        # Right arm is moving to new position
    LEFT_ARM_GRASPING = auto()       # Left arm is grasping the wall
    RIGHT_ARM_GRASPING = auto()      # Right arm is grasping the wall
    LEFT_ARM_RELEASING = auto()      # Left arm is releasing from wall
    RIGHT_ARM_RELEASING = auto()     # Right arm is releasing from wall
    TRANSITIONING = auto()           # Transitioning between walls
    GOAL_REACHED = auto()            # Goal position reached
    ERROR = auto()                   # Error state
    UNSCREWING = auto()              # Performing screw unscrewing task

class GripperState(Enum):
    """Gripper states"""
    OPEN = auto()
    CLOSED = auto()
    OPENING = auto()
    CLOSING = auto()

class CrawlerState:
    """Represents a state in the crawler's path (position + arm holding)"""
    def __init__(self, position, wall_normal, active_arm, anchored_arm_pos=None):
        self.position = position
        self.wall_normal = wall_normal
        self.active_arm = active_arm  # 'left' or 'right' (the arm that MOVES to this position)
        self.anchored_arm_pos = anchored_arm_pos # Position of the OTHER arm (anchor)
    
    def __repr__(self):
        return f"CrawlerState(pos={self.position}, arm={self.active_arm})"
