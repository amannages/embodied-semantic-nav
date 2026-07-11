# mapping/semantic_map.py
# Phase 4: Wiring Perception Pipeline from phase 2 to exploration loop from phase 3

import numpy as np

class SemanticMap:
    def __init__(self, width=100, height=100, resolution=0.25, origin=None):
        self.width = width
        self.height = height
        self.resolution = resolution
        self.origin = np.array(origin if origin is not None else [width // 2, height // 2])
        self.grid = {}
        self.sightings = {}

    def world_to_grid(self, x, z):
        col = int(round(x / self.resolution) + self.origin[0])
        row = int(round(z / self.resolution) + self.origin[1])
        row = int(np.clip(row, 0, self.height - 1))
        col = int(np.clip(col, 0, self.width - 1))
        return row, col

    def _get_cell(self, row, col):
        key = (row, col)
        if key not in self.grid:
            self.grid[key] = {}
        return self.grid[key]

    def update(self, detections, agent_x, agent_z):
        """
        Called after every YOLO detection pass.
        Stores each detection in the semantic map at the agent's current grid cell.
        """
        row, col = self.world_to_grid(agent_x, agent_z)
        cell = self._get_cell(row, col)

        for det in detections:
            label = det["label"]
            conf = float(det["confidence"])

            if label not in self.sightings:
                self.sightings[label] = []

            self.sightings[label].append({
                "position": (agent_x, agent_z),
                "confidence": conf,
                "row": row,
                "col": col,
            })

            if label not in cell:
                cell[label] = {"confidence": conf, "count": 1}
            else:
                prev_conf = cell[label]["confidence"]
                cell[label]["confidence"] = max(prev_conf, conf)
                cell[label]["count"] += 1

    def best_sighting(self, label):
        """
        Returns the highest-confidence sighting of a given label,
        or None if the object has never been seen.
        """
        if label not in self.sightings:
            return None
        return max(self.sightings[label], key=lambda s: s["confidence"])

    def get_cell_confidence(self, label, row, col):
        cell = self.grid.get((row, col), {})
        if label not in cell:
            return None
        return cell[label]["confidence"]

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
