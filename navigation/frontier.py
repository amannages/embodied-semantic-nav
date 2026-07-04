# navigation/frontier.py
# A frontier is any free cell that is adjacent to an unknown cell.
# The following algorithm is simple: always move towards the closest frontier
# and when you reach it, it becomes an "explored free space" and new frontiers
# appear beyond it. When there are no more frontiers, the exploration is complete.

import numpy as np
from navigation.occupancy_grid import OccupancyGrid, FREE, OCCUPIED, UNKNOWN

class FrontierExplorer:
    def __init__(self, occupancy_grid):
        """
        occupancy_grid: An instance of the OccupancyGrid class.
        Explorer reads from it to find frontiers
        Navigator writes to it as the robot moves
        They share the same occupancy grid so that 
        the robot can explore and navigate in a coordinated manner.
        """
        self.grid = occupancy_grid
    
    def find_frontiers(self):
        """
        A frotier cell is any FREE cell that has at least one UKNOWN neighbor.
        Check all 8 neigbors to catch unknown diagonals as well.
        Returns a list of (row, col) tuples of grid indices of frontier cells.
        """
        g = self.grid.grid
        size = self.grid.width * self.grid.height
        frontiers = []

        # vectorized approach to find frontiers
        free_rows, free_cols = np.where(g == FREE)

        for row, col in zip(free_rows, free_cols):
            # Check all 8 neighbors
            row_min = max(row - 1, 0)
            row_max = min(size - 1, row + 1)
            col_min = max(col - 1, 0)
            col_max = min(size - 1, col + 1)

            neighborhood = g[row_min:row_max + 1, col_min:col_max + 1]

            # Check if any neighbor is unknown
            if UNKNOWN in neighborhood:
                frontiers.append((row, col))
            
        return frontiers
    
    def nearest_frontier(self, agent_x, agent_z):
        """
        Find the nearest frontier cell to the robots current position (agent_x, agent_z) 
        in world coordinates. Uses euclidan distance in grid space.

        Returns (row, col) of the nearest frontier, or None if no frontiers exist.
        (exploration is complete)
        """
        frontiers = self.find_frontiers()
        if not frontiers:
            return None 
        
        agent_row, agent_col = self.grid.world_to_grid(agent_x, agent_z)
        agent_pos = np.array([agent_row, agent_col])

        frontier_array = np.array(frontiers)
        distances = np.linalg.norm(frontier_array - agent_pos, axis=1)
        nearest_index = np.argmin(distances)

        return frontiers[nearest_index]
    
    def frontier_count(self):
        """
        Returns the number of frontier cells in the occupancy grid.
        """
        return len(self.find_frontiers())