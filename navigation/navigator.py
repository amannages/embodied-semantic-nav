# navigation/navigator.py
# AI2-THOR's action space is discrete (MoveAhead = 0.25m forward, RotateLeft/Right = 90 degree turn)
# Navigator will answer two questions:
# 1. Am I facing the target direction?
# 2. Is there something blocking me?
# If yes to both, then move. If not facing right direction, then rotate. If something is blocking,
# mark as occupied, and then pick a new frontier.

import numpy as np
import math
from navigation.occupancy_grid import OccupancyGrid, FREE, OCCUPIED, UNKNOWN
from navigation.frontier import FrontierExplorer

# AI2-THOR rotation convention:
# 0°   = facing +z (forward in world space)
# 90°  = facing +x (right)
# 180° = facing -z (backward)
# 270° = facing -x (left)

# Direction Vectors for each heading
DIRECTION_VECTORS = {
    0:   np.array([0, 1]),   # +z
    90:  np.array([1, 0]),   # +x
    180: np.array([0, -1]),  # -z
    270: np.array([-1, 0])   # -x
}

class Navigator:
    def __init__(self, controller, grid_size = 100, grid_resolution = 0.25):
        """
        controller: An instance of the AI2-THOR controller.
        grid_size: The size of the occupancy grid (grid_size x grid_size).
        grid_resolution: The resolution of each cell in meters.
        """
        self.controller = controller
        self.occupancy_grid = OccupancyGrid(width=grid_size, height=grid_size, resolution=grid_resolution)
        self.frontier_explorer = FrontierExplorer(self.occupancy_grid)

        # Initialize the agent's position and orientation, gets updates after every action
        self.agent_x, self.agent_z = 0.0, 0.0  # Agent's position in world coordinates
        self.agent_yaw = 0.0  # Agent's orientation in degrees (0° = facing +z)


        self.step_count = 0

    #---------------------------------------------------------------------------
    # State Synchronization Functions
    #---------------------------------------------------------------------------

    def sync_from_event(self, event):
        """
        After every AI2-THOR action, call this to keep our internal state consistent
        with the simulator's ground truth.
        """
        position = event.metadata['agent']['position']
        rotation = event.metadata['agent']['rotation']

        self.agent_x = position['x']
        self.agent_z = position['z']
        self.agent_yaw = rotation['y'] % 360  # Normalize yaw to [0, 360)

        # Mark the current position as free in the occupancy grid
        self.occupancy_grid.mark_free(self.agent_x, self.agent_z)

    #---------------------------------------------------------------------------
    # Basic Movement Functions
    #---------------------------------------------------------------------------
    
    def move_ahead(self):
        """
        Move the robot ahead by 0.25m in the direction it is currently facing.
        """
        event = self.controller.step(action="MoveAhead")
        if event.metadata['lastActionSuccess']:
            self.sync_from_event(event)
        else:
            # If the move failed, something is blocking the robot.
            # So we mark the cell in front of the robot as occupied.
            dx, dz = DIRECTION_VECTORS[self.agent_yaw]
            blocked_x = self.agent_x + dx * self.occupancy_grid.resolution
            blocked_z = self.agent_z + dz * self.occupancy_grid.resolution
            self.occupancy_grid.mark_occupied(blocked_x, blocked_z)
        self.step_count += 1
        return event
    
    def rotate_left(self):
        """
        Rotate the robot left by 90 degrees.
        """
        event = self.controller.step(action="RotateLeft")
        if event.metadata['lastActionSuccess']:
            self.sync_from_event(event)
        self.step_count += 1
        return event
    
    def rotate_right(self):
        """
        Rotate the robot right by 90 degrees.
        """
        event = self.controller.step(action="RotateRight")
        if event.metadata['lastActionSuccess']:
            self.sync_from_event(event)
        self.step_count += 1
        return event
    
    #---------------------------------------------------------------------------
    # Rotation Optimization Functions
    #---------------------------------------------------------------------------   

    def face_direction(self, target_yaw):
        """
        Rotate the robot to face a specific yaw anle (0, 90, 180, 270 degrees).
        Choose the shortest rotation direction (left or right).
        """
        target_yaw = target_yaw % 360  # Normalize target yaw to [0, 360)
        
        while self.agent_yaw != target_yaw:
            # We calculate the diff in yaw
            diff = (target_yaw - self.agent_yaw) % 360
            if diff == 0:
                break # facing right direction
            elif diff <= 180:
                self.rotate_right()
            else:
                self.rotate_left()

    def face_toward(self, target_x, target_z):
        """
        Rotate the robot to face toward a specific world coordinate (target_x, target_z).
        """
        dx = target_x - self.agent_x
        dz = target_z - self.agent_z
        # atan gives us the angle in radians, we convert to degrees and normalize to [0, 360)
        angle = math.degrees(math.atan2(dx, dz)) % 360
        snapped = int(round(angle / 90) * 90) % 360  # Snap to nearest 90 degrees
        self.face_direction(snapped)

    #---------------------------------------------------------------------------
    # Scan Current Position Functions
    #---------------------------------------------------------------------------

    def scan_360(self):
        """
        Rotate the robot 360 degrees in 90 degree increments, scanning the environment.
        After each rotation, we mark the current position as free in the occupancy grid.
        Also returns all frames captured, which will be useful for integrating into YOLO.
        """
        frames = []
        for _ in range(4):
            event = self.rotate_right()
            self.occupancy_grid.mark_free(self.agent_x, self.agent_z)
            frames.append(event.frame)
        return frames
        