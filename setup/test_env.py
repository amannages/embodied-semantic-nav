from ai2thor.controller import Controller
import cv2

controller = Controller(
    agentMode="locobot",
    visibilityDistance=1.5,
    scene="FloorPlan1",
    gridSize=0.25,
    movementGaussianSigma=0.005,
    rotateStepDegrees=90,
    rotateGaussianSigma=0.5,
    renderDepthImage=False,
    renderInstanceSegmentation=False,
    width=300,
    height=300,
    fieldOfView=60
)
event = controller.step(action="RotateLeft")

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