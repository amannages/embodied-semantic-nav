# mapping/semantic_map.py
# Phase 4: Wiring Perception Pipeline from phase 2 to exploration loop from phase 3

import numpy as np

class SemanticMap:
    def __init__(self):
        # Stores all sightings: {label: [{"position": (x,z), "confidence": float}]}
        self.sightings = {}

    def update(self, detections, agent_x, agent_z):
        """
        Called after every YOLO detection pass.
        Stores each detection with the agent's position at time of sighting.
        """
        for det in detections:
            label = det["label"]
            conf = det["confidence"]

            if label not in self.sightings:
                self.sightings[label] = []

            self.sightings[label].append({
                "position":   (agent_x, agent_z),
                "confidence": conf
            })
        
    def best_sighting(self, label):
        """
        Returns the highest-confidence sighting of a given label,
        or None if the object has never been seen.
        """
        if label not in self.sightings:
            return None
        return max(self.sightings[label], key=lambda s: s["confidence"])

    def has_seen(self, label):
        return label in self.sightings

    def summary(self):
        print("\n=== SEMANTIC MAP ===")
        for label, sightings in sorted(self.sightings.items()):
            best = max(sightings, key=lambda s: s["confidence"])
            print(f"  {label:20s} | seen {len(sightings):3d}x | "
                  f"best conf: {best['confidence']:.2f} | "
                  f"best pos: ({best['position'][0]:.2f}, {best['position'][1]:.2f})")
        print("===================\n")
