# navigation/occupancy_grid.py
# start of phase 3: naviagtion. "Robot" can see but does not have a sense of place.
# We will build an occupancy grid to tell the robot what it has and hasnt seen, along
# with weither what it has seen is free or occupied.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches #this is for drawing rectangles on the plot

UNKNOWN = 0
FREE = 1
OCCUPIED = 2

class OccupancyGrid:
    def __init__(self, width=100, height=100, resolution=0.25):
        """
        width: number of cells in the x direction
        height: number of cells in the y direction
        resolution: size of each cell in meters in the real world

        With size = 100 and resolution = 0.25, the grid will cover 25m 
        by 25m, which should be more than enough for any FloorPlan.
        """
        self.width = width
        self.height = height
        self.resolution = resolution
        self.grid = np.full((height, width), UNKNOWN, dtype=np.uint8)  # Initialize grid with UNKNOWN

        # we would like our robot to start at the center of the grid
        self.origin = np.array([width // 2, height // 2])  

    # ---------------------------------------------------------------------------
    # Coordinate Conversion Functions
    # ---------------------------------------------------------------------------
    def world_to_grid(self, x, z):
        """
        Convert AI2-THOR world coordinates (x, z) to grid indices (row, col).

        Why x, z and not x, y? Because in AI2-THOR, 
        x = left/right
        y = up/down (ignoring it because we are going to do 2D mapping (imagine looking down from the ceiling))
        z = forward/backward
        """
        col = int(round(x / self.resolution) + self.origin[0])
        row = int(round(z / self.resolution) + self.origin[1])
        # clamp to grid boundaries so that we do not ever go outside of range
        row = np.clip(row, 0, self.height - 1)
        col = np.clip(col, 0, self.width - 1)
        return row, col
    
    def grid_to_world(self, row, col):
        """
        Convert grid indices (row, col) to AI2-THOR world coordinates (x, z).
        """
        x = (col - self.origin[0]) * self.resolution
        z = (row - self.origin[1]) * self.resolution
        return x, z
    
    #---------------------------------------------------------------------------
    # Marking Cells in the Grid Functions
    #---------------------------------------------------------------------------
    def mark_free(self, x, z):
        row, col = self.world_to_grid(x, z)
        self.grid[row, col] = FREE

    def mark_occupied(self, x, z):
        row, col = self.world_to_grid(x, z)
        self.grid[row, col] = OCCUPIED

    def mark_free_path(self, start_x, start_z, end_x, end_z):
        """
        Mark all cells along a straight line path as free.
        We will use Bresenham's line algorithm to determine which cells to mark.

        We can only use this when we know that everything between x_start, z_start and x_end, z_end is free. 
        This is the case when we have a depth camera and we are marking the path to an object that we can see.
        """
        start_row, start_col = self.world_to_grid(start_x, start_z)
        end_row, end_col = self.world_to_grid(end_x, end_z)

        # Bresenham's line algorithm
        cells = []
        diff_row = abs(end_row - start_row)
        diff_col = abs(end_col - start_col)
        row = start_row
        col = start_col
        step_row = 1 if start_row < end_row else -1 # else -1 in case we are going "backwards"
        step_col = 1 if start_col < end_col else -1
        error = diff_row - diff_col

        # Loop until we reach the end cell
        while True:
            cells.append((row, col))
            if row == end_row and col == end_col: # navigated to the end cell
                break
            error2 = error * 2
            if error2 > -diff_col:
                error -= diff_col
                row += step_row
            if error2 < diff_row:
                error += diff_row
                col += step_col

        for row, col in cells:
            if 0 <= row < self.height and 0 <= col < self.width:
                if self.grid[row, col] != OCCUPIED:  # Only mark as free if not already occupied
                    self.grid[row, col] = FREE

    def visualize(self):
        plt.imshow(self.grid, cmap='gray', origin='lower')
        plt.colorbar(ticks=[UNKNOWN, FREE, OCCUPIED], label='Occupancy State')
        plt.clim(-0.5, 2.5)
        plt.title('Occupancy Grid')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()