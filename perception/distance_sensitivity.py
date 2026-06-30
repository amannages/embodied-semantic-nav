from ai2thor.controller import Controller
from perception import ObjectDetector
import cv2
import os

os.makedirs("validation_frames", exist_ok=True)

detector = ObjectDetector(model_size="s", confidence_threshold=0.3)
controller = Controller(scene="FloorPlan1")

print("\n" + "="*60)
print("DISTANCE-SENSITIVITY TEST: Approaching the 270°")
print("="*60)

# (Adjust rotation or target object type if your specific scene layout differs)
controller.step("RotateLeft") 

for step in range(5):
    # Get the current frame
    event = controller.step("MoveAhead")
    frame = event.frame

    # Ground truth from ai2thor
    gt_visible = set([o["objectType"] for o in event.metadata["objects"] if o["visible"]])

    # YOLO detection from raw pixels
    detected_objects = detector.detect_objects(frame)
    yolo_visible = set([obj["label"] for obj in detected_objects])

    # save the annotated frame for visual inspection
    annotated_frame = detector.draw_bounding_boxes(frame, detected_objects)
    cv2.imwrite(f"validation_frames/fridge_step_{step+1}.png", annotated_frame)

    print(f"\n--- {step + 1} ---")
    print(f"  Ground truth:   {sorted(gt_visible)}")
    print(f"  YOLO detected:  {sorted(yolo_visible)}")
    print(f"  YOLO missed:    {sorted(gt_visible - yolo_visible)}")
    print(f"  YOLO invented:  {sorted(yolo_visible - gt_visible)}")  # false positives
    # At the bottom of each angle's print block, add:
    print(f"  Raw YOLO labels: {sorted(set(d['raw_label'] for d in detected_objects))}")

controller.stop()