from ai2thor.controller import Controller
import cv2
import os

os.makedirs("sweep_frames", exist_ok=True)

controller = Controller(scene="FloorPlan1")

all_visible = {}  # angle -> list of visible object types

for i, angle in enumerate([0, 90, 180, 270]):
    if i > 0:
        event = controller.step("RotateRight")
    else:
        event = controller.step("Pass")
    # rest of loop unchanged

    # Save frame
    frame = event.frame
    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    cv2.imwrite(f"sweep_frames/angle_{angle}.png", frame_bgr)

    # Log visible objects
    visible = [o["objectType"] for o in event.metadata["objects"] if o["visible"]]
    all_visible[angle] = visible
    print(f"\n--- {angle}° ---")
    print(f"Visible: {visible}")

print("\n=== FULL ROOM OBJECT INVENTORY ===")
seen_all = set()
for angle, objs in all_visible.items():
    seen_all.update(objs)
print(sorted(seen_all))

controller.stop()