import sys
import os 
sys.path.insert(0, os.path.dirname(__file__))

from ai2thor.controller import Controller
from navigation.navigator import Navigator
from mapping.semantic_map import SemanticMap
from perception.perception import ObjectDetector

# Create the mapping/ package
os.makedirs("mapping", exist_ok=True)
open("mapping/__init__.py", "w").close()

controller = Controller(
    scene="FloorPlan1",
    gridSize=0.25,          
    rotateStepDegrees=90,   
    renderDepthImage=False,
    renderInstanceSegmentation=False,
    timeout=300.0            # <-- Increase timeout window to 5 mins for the initial download
)

navigator = Navigator(controller, grid_size=100, grid_resolution=0.25)
detector = ObjectDetector(model_size="s", confidence_threshold=0.35)
sem_map = SemanticMap()

def on_step(event, nav):
    """Called after every move — run YOLO and update semantic map."""
    detections = detector.detect_objects(event.frame)
    sem_map.update(detections, nav.agent_x, nav.agent_z)

# Run the Exploration
occupancy_grid = navigator.explore(max_steps=400, on_step=on_step)

# Print what the Robot Learned
sem_map.summary()

# Save the final map
occupancy_grid.visualize(
    agent_pos=(navigator.agent_x, navigator.agent_z),
    save_path="explored_map.png"
)
print("Map saved to explored_map.png")

controller.stop()
