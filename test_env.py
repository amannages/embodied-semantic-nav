from ai2thor.controller import Controller

# Initialize the simulator in headless mode for testing
controller = Controller(scene="FloorPlan1", width=300, height=300)

# Move the robot agent forward
event = controller.step(action="MoveAhead")

# Print the final coordinates
print("Agent Position:", event.metadata["agent"]["position"])
controller.stop()
