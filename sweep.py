#performs a 360 degree sweep of the room and saves 4 images of the room.
from ai2thor.controller import Controller
import cv2
import os

os.makedirs("sweep_images", exist_ok=True)

controller = Controller(scene="FloorPlan1", agentMode="locobot", width=300, height=300)

all_visible = {}

for i, angle in enumerate([0, 90, 180, 270]):
    event = controller.step(action="RotateRight", degrees=angle)
    
    # Save the robot's current camera view
    frame = event.frame  # numpy array, shape (H, W, 3), RGB
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"sweep_images/view_{i}.png", frame_bgr)

    # Print position
    print(f"Position at angle {angle}:", event.metadata["agent"]["position"])
    print("Frame shape:", frame.shape)

    # Also print all visible objects right now
    visible = [o["objectType"] for o in event.metadata["objects"] if o["visible"]]
    all_visible[angle] = visible
    print(f"Visible objects at angle {angle}:", visible)

print("\n=== FULL ROOM OBJECT INVENTORY ===")
seen_objects = set()
for angle, objects in all_visible.items():
    for obj in objects:
        seen_objects.add(obj)

print("All seen objects:", sorted(seen_objects))