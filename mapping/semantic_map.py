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

    def grid_to_world(self, row, col):
        x = (col - self.origin[0]) * self.resolution
        z = (row - self.origin[1]) * self.resolution
        return x, z

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

    def _iter_label_cells(self, label):
        for (row, col), cell in self.grid.items():
            if label in cell:
                yield row, col, cell[label]

    def cells_containing(self, label):
        """
        Return all grid cells that have observed the given label.
        """
        cells = []
        for row, col, data in self._iter_label_cells(label):
            x, z = self.grid_to_world(row, col)
            cells.append({
                "row": row,
                "col": col,
                "position": (x, z),
                "confidence": data["confidence"],
                "count": data["count"],
            })
        return cells

    def nearest_cell_with_label(self, label, agent_x, agent_z):
        """
        Return the closest grid cell containing the label to the agent position.
        """
        cells = self.cells_containing(label)
        if not cells:
            return None

        agent_row, agent_col = self.world_to_grid(agent_x, agent_z)

        def cell_distance(cell):
            return (cell["row"] - agent_row) ** 2 + (cell["col"] - agent_col) ** 2

        best_cell = min(cells, key=cell_distance)
        best_cell = dict(best_cell)
        best_cell["distance_cells"] = float(cell_distance(best_cell) ** 0.5)
        return best_cell

    def best_known_location(self, label):
        """
        Return the strongest grid cell for a label based on confidence, then count.
        """
        cells = self.cells_containing(label)
        if not cells:
            return None

        best_cell = max(cells, key=lambda cell: (cell["confidence"], cell["count"]))
        return best_cell

    def get_cell_confidence(self, label, row, col):
        cell = self.grid.get((row, col), {})
        if label not in cell:
            return None
        return cell[label]["confidence"]

    def has_seen(self, label):
        return label in self.sightings
    
    def reliable_labels(self, min_confidence=0.7, min_sightings=5):
        """
        Returns only labels the robot is genuinely confident about.
        Filters out hallucinations from simulator geometry.
        """
        reliable = {}
        for label, sightings in self.sightings.items():
            best_conf = max(s["confidence"] for s in sightings)
            if best_conf >= min_confidence and len(sightings) >= min_sightings:
                reliable[label] = {
                    "best_confidence": best_conf,
                    "sighting_count": len(sightings),
                    "best_position": max(sightings, 
                        key=lambda s: s["confidence"])["position"]
                }
        return reliable

    def summary(self):
        print("\n=== SEMANTIC MAP ===")
        for label, sightings in sorted(self.sightings.items()):
            best = max(sightings, key=lambda s: s["confidence"])
            print(f"  {label:20s} | seen {len(sightings):3d}x | "
                  f"best conf: {best['confidence']:.2f} | "
                  f"best pos: ({best['position'][0]:.2f}, {best['position'][1]:.2f})")
        print("===================\n")
