import sys
import os 
sys.path.insert(0, os.path.dirname(__file__))

from ai2thor.controller import Controller
from ai2thor.platform import CloudRendering  # <-- Import the CloudRendering platform
from navigation.navigator import Navigator

controller = Controller(
    scene="FloorPlan1",
    gridSize=0.25,          
    rotateStepDegrees=90,   
    renderDepthImage=False,
    renderInstanceSegmentation=False,
    timeout=300.0            # <-- Increase timeout window to 5 mins for the initial download
)

navigator = Navigator(controller, grid_size=100, grid_resolution=0.25)

# Run the Exploration
occupancy_grid = navigator.explore(max_steps=400)

# Save the final map
occupancy_grid.visualize(
    agent_pos=(navigator.agent_x, navigator.agent_z),
    save_path="explored_map.png"
)
print("Map saved to explored_map.png")

controller.stop()
