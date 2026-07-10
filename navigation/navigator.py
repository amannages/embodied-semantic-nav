# navigation/navigator.py
# AI2-THOR's action space is discrete (MoveAhead = 0.25m forward, RotateLeft/Right = 90 degree turn)
# Navigator will answer two questions:
# 1. Am I facing the target direction?
# 2. Is there something blocking me?
# If yes to both, then move. If not facing right direction, then rotate. If something is blocking,
# mark as occupied, and then pick a new frontier.

import numpy as np
import math
from navigation.occupancy_grid import OccupancyGrid, FREE, OCCUPIED, UNKNOWN
from navigation.frontier import FrontierExplorer

# AI2-THOR rotation convention:
# 0°   = facing +z (forward in world space)
# 90°  = facing +x (right)
# 180° = facing -z (backward)
# 270° = facing -x (left)

# Direction Vectors for each heading
DIRECTION_VECTORS = {
    0:   np.array([0, 1]),   # +z
    90:  np.array([1, 0]),   # +x
    180: np.array([0, -1]),  # -z
    270: np.array([-1, 0])   # -x
}

class Navigator:
    def __init__(self, controller, grid_size = 100, grid_resolution = 0.25):
        """
        controller: An instance of the AI2-THOR controller.
        grid_size: The size of the occupancy grid (grid_size x grid_size).
        grid_resolution: The resolution of each cell in meters.
        """
        self.controller = controller
        self.occupancy_grid = OccupancyGrid(width=grid_size, height=grid_size, resolution=grid_resolution)
        self.frontier_explorer = FrontierExplorer(self.occupancy_grid)

        # Initialize the agent's position and orientation, gets updates after every action
        self.agent_x, self.agent_z = 0.0, 0.0  # Agent's position in world coordinates
        self.agent_yaw = 0.0  # Agent's orientation in degrees (0° = facing +z)

        self.step_count = 0
        self.last_frontier = None

    #---------------------------------------------------------------------------
    # State Synchronization Functions
    #---------------------------------------------------------------------------

    def sync_from_event(self, event):
        """
        After every AI2-THOR action, call this to keep our internal state consistent
        with the simulator's ground truth.
        """
        position = event.metadata['agent']['position']
        rotation = event.metadata['agent']['rotation']

        self.agent_x = position['x']
        self.agent_z = position['z']
        self.agent_yaw = int(round(rotation['y'])) % 360

        self.occupancy_grid.mark_free(self.agent_x, self.agent_z)
        self.update_map_from_event(event) 

    #---------------------------------------------------------------------------
    # Basic Movement Functions
    #---------------------------------------------------------------------------
    
    def move_ahead(self):
        """
        Move the robot ahead by 0.25m in the direction it is currently facing.
        """
        event = self.controller.step(action="MoveAhead")
        if event.metadata['lastActionSuccess']:
            self.sync_from_event(event)
        else:
            # If the move failed, something is blocking the robot.
            # So we mark the cell in front of the robot as occupied.
            dx, dz = DIRECTION_VECTORS[int(self.agent_yaw)]
            blocked_x = self.agent_x + dx * self.occupancy_grid.resolution
            blocked_z = self.agent_z + dz * self.occupancy_grid.resolution
            self.occupancy_grid.mark_occupied(blocked_x, blocked_z)
        self.step_count += 1
        return event
    
    def rotate_left(self):
        """
        Rotate the robot left by 90 degrees.
        """
        event = self.controller.step(action="RotateLeft")
        if event.metadata['lastActionSuccess']:
            self.sync_from_event(event)
        self.step_count += 1
        return event
    
    def rotate_right(self):
        """
        Rotate the robot right by 90 degrees.
        """
        event = self.controller.step(action="RotateRight")
        if event.metadata['lastActionSuccess']:
            self.sync_from_event(event)
        self.step_count += 1
        return event
    
    #---------------------------------------------------------------------------
    # Rotation Optimization Functions
    #---------------------------------------------------------------------------   

    def face_direction(self, target_yaw):
        """
        Rotate the robot to face a specific yaw anle (0, 90, 180, 270 degrees).
        Choose the shortest rotation direction (left or right).
        """
        target_yaw = target_yaw % 360  # Normalize target yaw to [0, 360)
        
        while self.agent_yaw != target_yaw:
            # We calculate the diff in yaw
            diff = (target_yaw - self.agent_yaw) % 360
            if diff == 0:
                break # facing right direction
            elif diff <= 180:
                self.rotate_right()
            else:
                self.rotate_left()

    def face_toward(self, target_x, target_z):
        """
        Rotate the robot to face toward a specific world coordinate (target_x, target_z).
        """
        dx = target_x - self.agent_x
        dz = target_z - self.agent_z
        # atan gives us the angle in radians, we convert to degrees and normalize to [0, 360)
        angle = math.degrees(math.atan2(dx, dz)) % 360
        snapped = int(round(angle / 90) * 90) % 360  # Snap to nearest 90 degrees
        self.face_direction(snapped)

    #---------------------------------------------------------------------------
    # Scan Current Position Functions
    #---------------------------------------------------------------------------

    def scan_360(self):
        """
        Rotate the robot 360 degrees in 90 degree increments, scanning the environment.
        After each rotation, we mark the current position as free in the occupancy grid.
        Also returns all frames captured, which will be useful for integrating into YOLO.
        """
        frames = []
        for _ in range(4):
            event = self.rotate_right()
            self.update_map_from_event(event)   
            frames.append(event.frame)
        return frames
    
    # -------------------------------------------------------------------------
    # High-level Functions: move toward a grid cell and run the frontier exploration
    # -------------------------------------------------------------------------

    def move_toward_cell(self, target_row, target_col):
        """
        Take one step toward a target grid cell.
        Faces toward it, then moves ahead.
        Returns True if move succeeded.
        """
        target_x, target_z = self.occupancy_grid.grid_to_world(target_row, target_col)
        self.face_toward(target_x, target_z)
        event = self.move_ahead()
        return event.metadata["lastActionSuccess"]
    
    def update_map_from_event(self, event):
        """
        Use AI2-THOR's visibility metadata to mark free space via ray-casting.
        Every visible object tells us the line-of-sight from agent to object is free.
        """
        agent_x = self.agent_x
        agent_z = self.agent_z

        for obj in event.metadata["objects"]:
            if not obj["visible"]:
                continue

            obj_pos = obj.get("position")
            if obj_pos is None:
                continue

            obj_x = obj_pos["x"]
            obj_z = obj_pos["z"]

            # Ray-cast: mark everything between agent and object as free
            self.occupancy_grid.mark_free_path(agent_x, agent_z, obj_x, obj_z)

        # Also mark current cell explicitly
        self.occupancy_grid.mark_free(agent_x, agent_z)
    
    def find_path_bfs(self, target_row, target_col):
        """
        BFS through non-occupied cells to find a walkable path from current
        position to target. Navigates through both FREE and UNKNOWN cells —
        unknown doesn't mean blocked, it just means unvisited.
        
        Returns: list of (row, col) cells to visit (excluding start cell),
                or None if the target is completely unreachable.
        """
        from collections import deque

        start_row, start_col = self.occupancy_grid.world_to_grid(
            self.agent_x, self.agent_z
        )

        if start_row == target_row and start_col == target_col:
            return []

        height = self.occupancy_grid.height
        width  = self.occupancy_grid.width
        grid   = self.occupancy_grid.grid

        # visited maps cell → parent cell, so we can reconstruct the path
        visited = {(start_row, start_col): None}
        queue   = deque([(start_row, start_col)])

        while queue:
            r, c = queue.popleft()

            if r == target_row and c == target_col:
                # Reconstruct path by walking parent pointers back to start
                path = []
                current = (r, c)
                while visited[current] is not None:
                    path.append(current)
                    current = visited[current]
                path.reverse()  # start → target order
                return path

            # Only 4-directional moves — matches robot's discrete action space
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                nr, nc = r + dr, c + dc
                if (0 <= nr < height and
                    0 <= nc < width  and
                    (nr, nc) not in visited and
                    grid[nr, nc] != OCCUPIED):   # can traverse FREE or UNKNOWN
                    visited[(nr, nc)] = (r, c)
                    queue.append((nr, nc))

        return None  # target is surrounded by occupied cells — truly unreachable

    # Replace exact equality checks with distance tolerance
    def _at_frontier(self, fr, fc, tolerance=1.5):
        cur_r, cur_c = self.occupancy_grid.world_to_grid(
            self.agent_x, self.agent_z
        )
        return math.sqrt((cur_r - fr)**2 + (cur_c - fc)**2) < tolerance

    def count_unknown_neighbors(self, row, col):
        grid = self.occupancy_grid.grid
        height = self.occupancy_grid.height
        width = self.occupancy_grid.width
        count = 0
        for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
            nr, nc = row + dr, col + dc
            if 0 <= nr < height and 0 <= nc < width and grid[nr, nc] == UNKNOWN:
                count += 1
        return count

    def explore(self, max_steps=200, on_step=None):
        import numpy as np

        # Initialize
        event = self.controller.step("Pass")
        self.sync_from_event(event)
        self.occupancy_grid.mark_free(self.agent_x, self.agent_z)

        print("Initial 360° scan...")
        self.scan_360()
        print(f"Starting exploration at ({self.agent_x:.2f}, {self.agent_z:.2f})")

        blacklisted = set()

        while self.step_count < max_steps:
            # --- Pick a frontier ---
            all_frontiers  = self.frontier_explorer.find_frontiers()
            valid_frontiers = [f for f in all_frontiers if f not in blacklisted]

            if not valid_frontiers:
                print("✅ Exploration complete — no reachable frontiers remain.")
                break

            # Sort by distance, then choose the most informative reachable frontier
            agent_row, agent_col = self.occupancy_grid.world_to_grid(
                self.agent_x, self.agent_z
            )
            agent_pos_arr = np.array([agent_row, agent_col])
            frontier_arr  = np.array(valid_frontiers)
            dists = np.linalg.norm(frontier_arr - agent_pos_arr, axis=1)
            sorted_indices = np.argsort(dists)

            scored_frontiers = []
            for idx in sorted_indices:
                fr, fc = valid_frontiers[idx]
                if dists[idx] < 1.5:
                    continue
                if self.last_frontier is not None and (fr, fc) == self.last_frontier:
                    continue

                path = self.find_path_bfs(fr, fc)
                if path is None or len(path) == 0:
                    continue

                score = (
                    self.count_unknown_neighbors(fr, fc),
                    -len(path),
                    -dists[idx]
                )
                scored_frontiers.append((score, fr, fc, path))

            if not scored_frontiers:
                # If every frontier was skipped due to last_frontier or distance, accept the nearest reachable frontier.
                for idx in sorted_indices:
                    fr, fc = valid_frontiers[idx]
                    if dists[idx] < 1.5:
                        continue
                    path = self.find_path_bfs(fr, fc)
                    if path is not None and len(path) > 0:
                        scored_frontiers.append(((0, -len(path), -dists[idx]), fr, fc, path))
                        break

            if not scored_frontiers:
                print("No BFS-reachable frontiers remain.")
                break

            scored_frontiers.sort(reverse=True)
            _, fr, fc, chosen_path = scored_frontiers[0]
            self.last_frontier = (fr, fc)
            fx, fz = self.occupancy_grid.grid_to_world(fr, fc)

            print(f"Step {self.step_count:3d} | "
                f"Agent: ({self.agent_x:.2f}, {self.agent_z:.2f}) | "
                f"→ Frontier: ({fx:.2f}, {fz:.2f}) | "
                f"Path: {len(chosen_path)} steps | "
                f"Valid: {len(valid_frontiers)} | "
                f"Blacklisted: {len(blacklisted)}")

            # --- Follow the BFS path step by step ---
            reached = False
            path_index = 0
            while path_index < len(chosen_path) and self.step_count < max_steps:
                step_r, step_c = chosen_path[path_index]

                cur_r, cur_c = self.occupancy_grid.world_to_grid(
                    self.agent_x, self.agent_z
                )
                if cur_r == fr and cur_c == fc:
                    reached = True
                    break

                success = self.move_toward_cell(step_r, step_c)

                if on_step is not None:
                    ev = self.controller.step("Pass")
                    on_step(ev, self)

                if not success:
                    new_path = self.find_path_bfs(fr, fc)
                    if new_path is None:
                        print(f"Blacklisting ({fx:.2f}, {fz:.2f}) - blocked after replan")
                        blacklisted.add((fr, fc))
                        self.occupancy_grid.mark_occupied(fx, fz)
                        break
                    chosen_path = new_path
                    path_index = 0
                    continue

                path_index += 1

            if not reached:
                cur_r, cur_c = self.occupancy_grid.world_to_grid(
                    self.agent_x, self.agent_z
                )
                if cur_r == fr and cur_c == fc:
                    reached = True

            if reached:
                self.scan_360()

            if not reached and (fr, fc) not in blacklisted:
                print(f"Blacklisting ({fx:.2f}, {fz:.2f}) - path exhausted")
                blacklisted.add((fr, fc))
                self.occupancy_grid.mark_occupied(fx, fz)

        print(f"Exploration ended after {self.step_count} steps.")
        return self.occupancy_grid

        