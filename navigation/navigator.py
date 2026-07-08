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