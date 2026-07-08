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
    
