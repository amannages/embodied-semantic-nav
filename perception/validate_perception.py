# validation script to compare yolo detections with ai2thor visible objects
from ai2thor.controller import Controller
from perception import ObjectDetector
import cv2
import os

os.makedirs("validation_frames", exist_ok=True)

detector = ObjectDetector(model_size="s", confidence_threshold=0.3)
controller = Controller(scene="FloorPlan1")

print("="*60)
print("PERCEPTION VALIDATION: YOLO vs Ground Truth")
print("="*60)

for i, angle in enumerate([0, 90, 180, 270]):
    if i > 0:
        event = controller.step("RotateRight")
    else:
        event = controller.step("Pass")

    # Get the current frame
    frame = event.frame

    # Ground truth from ai2thor
    gt_visible = set([o["objectType"] for o in event.metadata["objects"] if o["visible"]])

    # YOLO detection from raw pixels
    detected_objects = detector.detect_objects(frame)
    yolo_visible = set([obj["label"] for obj in detected_objects])

    # save the annotated frame for visual inspection
    annotated_frame = detector.draw_bounding_boxes(frame, detected_objects)
    cv2.imwrite(f"validation_frames/angle_{angle}.png", annotated_frame)

    print(f"\n--- {angle}° ---")
    print(f"  Ground truth:   {sorted(gt_visible)}")
    print(f"  YOLO detected:  {sorted(yolo_visible)}")
    print(f"  YOLO missed:    {sorted(gt_visible - yolo_visible)}")
    print(f"  YOLO invented:  {sorted(yolo_visible - gt_visible)}")  # false positives

controller.stop()