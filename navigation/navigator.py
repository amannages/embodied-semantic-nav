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