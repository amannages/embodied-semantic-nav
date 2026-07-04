# navigation/occupancy_grid.py
# start of phase 3: naviagtion. "Robot" can see but does not have a sense of place.
# We will build an occupancy grid to tell the robot what it has and hasnt seen, along
# with weither what it has seen is free or occupied.

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches #this is for drawing rectangles on the plot

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

    #---------------------------------------------------------------------------
    # Querying Cells in the Grid Functions
    #---------------------------------------------------------------------------
    def is_free(self, x, z):
        row, col = self.world_to_grid(x, z)
        return self.grid[row, col] == FREE

    def is_occupied(self, x, z):
        row, col = self.world_to_grid(x, z)
        return self.grid[row, col] == OCCUPIED

    def get_cell(self, x, z):
        row, col = self.world_to_grid(x, z)
        return self.grid[row, col]
    
    #---------------------------------------------------------------------------
    # Visualization Function
    #---------------------------------------------------------------------------
    def visualize(self, agent_pos=None, path=None, save_path=None):
        """
        Function to render the occupancy grid as a color image.
        Color Code:
        - Black: Unknown
        - White: Free
        - Red: Occupied
        - Blue: Robot's current position
        - Yellow: Planned Path
        """

        # Build RGB image from the occupancy grid
        rgb = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        rgb[self.grid == UNKNOWN] = [0, 0, 0]       # Black for unknown
        rgb[self.grid == FREE] = [255, 255, 255]  # White for free
        rgb[self.grid == OCCUPIED] = [255, 0, 0]    # Red for occupied

        # Draw planned path (if it exists)
        if path is not None:
            for (row, col) in path:
                if 0 <= row < self.height and 0 <= col < self.width:
                    rgb[row, col] = [255, 255, 0] # yellow = planned path

        # Draw agent's current position (if it exists)
        if agent_pos is not None:
            agent_row, agent_col = self.world_to_grid(*agent_pos)
            # Draw a small cross for the agent
            for dr in range(-2, 3):
                for dc in range(-2, 3):
                    nr, nc = agent_row + dr, agent_col + dc
                    if 0 <= nr < self.height and 0 <= nc < self.width:
                        rgb[nr, nc] = [0, 0, 255] # blue = agent's position

        # Display the image
        figure, axis = plt.subplots(1, 1, figsize=(8, 8))
        axis.imshow(rgb, origin='upper')
        axis.set_title("Occupancy Grid")
        axis.axis('off')

        # Build Legend
        patches = [
            mpatches.Patch(color='black', label='Unknown'),
            mpatches.Patch(color='white', label='Free'),
            mpatches.Patch(color='red', label='Occupied'),
            mpatches.Patch(color='blue', label="Agent's Position"),
            mpatches.Patch(color='yellow', label='Planned Path')
        ]
        if path:
            patches.append(mpatches.Patch(color=[1, 0.86, 0], label='Planned Path'))  # Yellow for planned path
        axis.legend(handles=patches, loc='upper right', fontsize=8)

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close()
        else:
            plt.show()
