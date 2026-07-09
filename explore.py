# explore.py
import sys
import os 
sys.path.insert(0, os.path.dirname(__file__))

from ai2thor.controller import Controller
from navigation.navigator import Navigator

controller = Controller(
    scene="FloorPlan1",
    gridSize=0.25,          # match our resolution
    rotateStepDegrees=90,   # match our rotation assumption
    renderDepthImage=False,
    renderInstanceSegmentation=False,
)

navigator = Navigator(controller, grid_size=100, resolution=0.25)


# Run the Exploration
occupancy_grid = navigator.explore(max_steps=150)

# Save the final map
occupancy_grid.visualize(
    agent_pos=(navigator.agent_x, navigator.agent_z),
    save_path="explored_map.png"
)
print("Map saved to explored_map.png")

controller.stop()