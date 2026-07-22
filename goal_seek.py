# goal_seek.py (put everything together)
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from ai2thor.controller import Controller
from navigation.navigator import Navigator
from mapping.semantic_map import SemanticMap
from perception.perception import ObjectDetector
from language.clip_resolver import CLIPResolver
from language.goal_navigator import GoalNavigator

controller = Controller(
    scene="FloorPlan1",
    gridSize=0.25,
    rotateStepDegrees=90,
    renderDepthImage=False,
    renderInstanceSegmentation=False,
    timeout=300.0
)

# Initialize All Components
navigator = Navigator(controller, grid_size=100, grid_resolution=0.25)
detector = ObjectDetector(model_size="s", confidence_threshold=0.35)
semantic_map = SemanticMap()
clip_resolver = CLIPResolver()
goal_navigator = GoalNavigator(navigator, semantic_map, clip_resolver, detector)

# Phase 1: Explore and Build Semantic Map
print("="*60)
print("PHASE 1: Exploring environment...")
print("="*60)

def on_step(event, nav):
    """Called after every move - run YOLO and update semantic map."""
    detections = detector.detect_objects(event.frame)
    semantic_map.update(detections, nav.agent_x, nav.agent_z)

navigator.explore(max_steps=400, on_step=on_step)
semantic_map.summary()

# Phase 2: Accept Natural Language Goal and Resolve to Semantic Label
print("="*60)
print("PHASE 2: Resolving natural language goal...")
print("="*60)

# user_query = input("Enter your goal (e.g., 'Find a microwave'): ")
# For now, we wiill loop through a few example queries
# Test queries — from literal to semantic
queries = [
    "a refrigerator",                    # direct — tests basic lookup
    "something I can drink from",        # semantic — should resolve to Mug
    "something used to cook food",       # semantic — should resolve to Microwave/StoveKnob
    "a household appliance",             # broad — tests CLIP's generalization
]

for user_query in queries:
    goal_navigator.execute_goal(user_query)

controller.stop()
