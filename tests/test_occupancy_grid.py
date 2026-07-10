import numpy as np

from navigation.occupancy_grid import OccupancyGrid, FREE, OCCUPIED, UNKNOWN


def test_mark_free_path_does_not_mark_endpoint_as_free():
    grid = OccupancyGrid(width=20, height=20, resolution=1.0)

    grid.mark_free_path(0, 0, 2, 0)

    start_row, start_col = grid.world_to_grid(0, 0)
    end_row, end_col = grid.world_to_grid(2, 0)

    assert grid.grid[start_row, start_col] == FREE
    assert grid.grid[end_row, end_col] != FREE
