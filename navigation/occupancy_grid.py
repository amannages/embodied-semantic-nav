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

    def update(self, x, y, state):
        if 0 <= x < self.width and 0 <= y < self.height:
            self.grid[y, x] = state

    def visualize(self):
        plt.imshow(self.grid, cmap='gray', origin='lower')
        plt.colorbar(ticks=[UNKNOWN, FREE, OCCUPIED], label='Occupancy State')
        plt.clim(-0.5, 2.5)
        plt.title('Occupancy Grid')
        plt.xlabel('X')
        plt.ylabel('Y')
        plt.show()