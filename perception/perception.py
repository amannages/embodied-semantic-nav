#Object Detection with Yolov8
#Ai2thor gave an overall view of a room, since it knew every object in the scene.
#However, in a real-world scenario, the robot will not have access to the entire scene 
#and will only be able to see what is in front of it. Therefore, we need to use object 
#detection to identify objects in the robot's current view.
#Validation step: we will compare the objects detected by Yolov8 with the 
#objects that Ai2thor says are visible in the current view.

#detector class
from ultralytics import YOLO
import cv2
import numpy as np

# Maps COCO labels → semantic concepts
# Left side: what YOLO outputs
# Right side: what your robot's semantic map will store
LABEL_MAP = {
    # (mapped_name, reliability)
    # reliability: 'high' = trust single detection
    #              'low'  = require multi-frame confirmation

    "microwave":    ("Microwave",    "high"),
    "oven":         ("StoveKnob",    "high"),
    "toaster":      ("Toaster",      "high"),
    "refrigerator": ("Fridge",       "high"),
    "sink":         ("Sink",         "high"),
    "potted plant": ("HousePlant",   "high"),
    "couch":        ("Sofa",         "high"),
    "chair":        ("Chair",        "high"),
    "bed":          ("Bed",          "high"),
    "dining table": ("DiningTable",  "high"),
    "tv":           ("Television",   "high"),

    "apple":        ("Apple",        "low"),
    "orange":       ("Orange",       "low"),
    "banana":       ("Banana",       "low"),
    "bowl":         ("Bowl",         "low"),
    "cup":          ("Mug",          "low"),   # YOLO calls mugs 'cup'
    "bottle":       ("Bottle",       "low"),
    "knife":        ("ButterKnife",  "low"),
    "book":         ("Book",         "low"),
    "vase":         ("Vase",         "low"),

    # Known hallucinations on simulator geometry — drop entirely
    "airplane":     (None, None),
    "suitcase":     (None, None),
    "train":        (None, None),
    "boat":         (None, None),
    "sports ball":  (None, None),
}

class ObjectDetector:
    def __init__(self, model_size = "s", confidence_threshold = 0.3):
        """
        model_size: 'n', 's', 'm', 'l', 'x'
        conf_threshold: minimum confidence to count as a detection
        """
        self.model = YOLO(f"yolov8{model_size}.pt")
        self.confidence_threshold = confidence_threshold

    def detect_objects(self, image):
        """
        image: numpy array (H, W, 3) in RGB (this is exactly what ai2thor gives us)
        return: list of detected objects with keys (label, confidence, bbox)
        """
        results = self.model(image, verbose=False)[0] #verbose = false to avoid per-frame printing
        detected_objects = []


        for box in results.boxes:
            confidence = float(box.conf[0])
            if confidence < self.confidence_threshold:
                continue

            raw_label = results.names[int(box.cls[0])]

            if raw_label in LABEL_MAP:
                mapped, reliability = LABEL_MAP[raw_label]
                if mapped is None: 
                    continue  # ignore this label
                label = mapped
            else:
                label = raw_label  # keep the original label if not in the map    
                reliability = "low"  # default to low reliability for unmapped labels
            # box.xyxy gives [x1, y1, x2, y2] in pixel coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detected_objects.append({
                "label": label,
                "raw_label": raw_label,
                "confidence": confidence,
                "reliability": reliability,
                "bbox": [x1, y1, x2, y2]
            })
        
        return detected_objects
    
    def draw_bounding_boxes(self, image, detected_objects):
        """
        image: numpy array (H, W, 3) in RGB
        detected_objects: list of detected objects with keys (label, confidence, bbox)
        return: BGR image (ready for cv2.imwrite)
        """
        frame_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        for obj in detected_objects:
            x1, y1, x2, y2 = obj["bbox"]
            label = obj["label"]
            confidence = obj["confidence"]

            # Draw rectangle
            cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Draw label backgrounde
            text = f"{label} {confidence:.2f}"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame_bgr, (x1, y1 - th - 4), (x1 + tw, y1), (0, 255, 0), -1)
            # Draw the label text
            cv2.putText(frame_bgr, text, (x1, y1 - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        return frame_bgr