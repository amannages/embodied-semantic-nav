from ai2thor.controller import Controller
import cv2

controller = Controller(scene="FloorPlan1")
event = controller.step(action="MoveAhead")

# Save the robot's current camera view
frame = event.frame  # numpy array, shape (H, W, 3), RGB
frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
cv2.imwrite("view.png", frame_bgr)

# Print position
print("Position:", event.metadata["agent"]["position"])
print("Frame shape:", frame.shape)

# Also print all visible objects right now
visible = [o["objectType"] for o in event.metadata["objects"] if o["visible"]]
print("Visible objects:", visible)