# language/goal_navigator.py
# Phase 5: Goal-Oriented Navigation
# Full Pipeline: Natural Language -> CLIP -> Semantic Map -> BFS -> Occupancy Grid -> Navigation

import math
import numpy as np

class GoalNavigator:
    def __init__(self, navigator, semantic_map, clip_resolver, detector):
        """
        navigator: An instance of the Navigator class for pathfinding and movement.
        semantic_map: An instance of the SemanticMap class for storing object detections.
        clip_resolver: An instance of the CLIPResolver class for resolving natural language queries to semantic labels.
        detector: An instance of the ObjectDetector class for detecting objects in the environment.
        """
        self.nav = navigator
        self.semantic_map = semantic_map
        self.clip = clip_resolver
        self.detector = detector

    #---------------------------------------------------------------------------
    # Step 1: Resolve Natural Language Query to Semantic Label
    #---------------------------------------------------------------------------

    def resolve_goal(self, user_query):
        """
        Use CLIP to map natural language goal to the best label in
        the semantic map.

        Will only consider labels that have been seen in the semantic map.
        """
        known_labels = list(self.semantic_map.reliable_labels(
            min_confidence=0.7, 
            min_sightings=5
        ))

        if not known_labels:
            print("Semantic map is empty. Please explore first.")
            return None, 0.0
        
        best_label, score, ranking = self.clip.resolve_label(
            user_query, known_labels, verbose=True
        )

        return best_label, score
    
    #---------------------------------------------------------------------------
    # Step 2: Find Best Known Position for a Label
    #---------------------------------------------------------------------------

    def locate_target(self, label):
        """
        Query the semantic map for the best known position of the given label.
        Uses nearest_cell_with_label to find the most reliable location relative to the 
        agent's current position.
        """
        cell = self.semantic_map.nearest_cell_with_label(
            label, 
            agent_x=self.nav.agent_x, 
            agent_z=self.nav.agent_z
        )

        if cell is None:
            print(f"No known location for label '{label}' in the semantic map.")
            return None
        
        print(f"\nTarget '{label}' last seen at "
              f"({cell['position'][0]:.2f}, {cell['position'][1]:.2f}) "
              f"with confidence {cell['confidence']:.2f}")

        return cell
    
    #---------------------------------------------------------------------------
    # Step 3: Navigate to Target Position
    #---------------------------------------------------------------------------

    def navigate_to_cell(self, target_row, target_col, target_label, max_steps=100):
        """
        BFS navigation to the target cell in the occupancy grid.
        Returns True if the target was reached, False otherwise.
        """
        print(f"\nNavigating to '{target_label}' at grid cell ({target_row}, {target_col})...")

        steps = 0
        while steps < max_steps:
            # Check if we are close enough to the target cell (or already there)
            current_row, current_col = self.nav.occupancy_grid.world_to_grid(self.nav.agent_x, self.nav.agent_z)
            distance = math.sqrt((current_row - target_row) ** 2 + (current_col - target_col) ** 2)

            if distance < 2.0:
                print(f"Close enough to target cell ({target_row}, {target_col}). Stopping navigation.")
                break

            # Use BFS to find the next step towards the target cell
            path = self.nav.find_path_bfs(target_row, target_col)
            if path is None:
                print("No path found to the target cell. Stopping navigation.")
                return False
            
            # Move to the next cell in the path
            for step_row, step_col in path:
                if steps >= max_steps:
                    print("Reached maximum navigation steps. Stopping.")
                    break

                # Check if target is already visible from current position
                event = self.nav.controller.step(action="Pass")  # Get current frame
                frame = event.frame
                detections = self.detector.detect_objects(frame)
                labels_in_view = [d["label"] for d in detections]

                if target_label in labels_in_view:
                    confidence = max(
                        d["confidence"] for d in detections 
                        if d["label"] == target_label
                    )
                    print(f"Target '{target_label}' detected in view with confidence {confidence:.2f}. Stopping navigation.")
                    return True
                
                # Move to the next cell
                success = self.nav.move_toward_cell(step_row, step_col)
                steps += 1

                if not success:
                    break # Replan on Next Outer Loop Iteration

        # Final check if target is visible after navigation
        print("At final position, checking for target visibility...")
        frames = self.nav.scan_360()

        for frame in frames:
            detections = self.detector.detect_objects(frame)
            labels_in_view = [d["label"] for d in detections]

            if target_label in labels_in_view:
                confidence = max(
                    d["confidence"] for d in detections 
                    if d["label"] == target_label
                )
                print(f"Target '{target_label}' detected in view with confidence {confidence:.2f}.")
                return True
            
        print(f" Reached final position but target '{target_label}' not detected. Navigation complete.")
        return False
    
    #---------------------------------------------------------------------------
    # Full Pipeline: Resolve Goal -> Locate Target -> Navigate
    #---------------------------------------------------------------------------

    def execute_goal(self, user_query):
        """
        Full pipeline to execute a natural language goal:
        1. Resolve the goal to a semantic label using CLIP.
        2. Locate the best known position of that label in the semantic map.
        3. Navigate to that position using BFS in the occupancy grid.
        4. Visual confirmation of the target object if detected.
        """
        print(f"\n{'='*60}")
        print(f"GOAL: '{user_query}'")
        print(f"{'='*60}")

        # Step 1: Resolve Goal
        label, score = self.resolve_goal(user_query)
        if label is None:
            print("Could not resolve goal. Exiting.")
            return False
        
        print(f"\nResolved to: '{label}' (CLIP score: {score:.3f})")
        
        # Step 2: Locate Target
        cell = self.locate_target(label)
        if cell is None:
            print("Could not locate target. Exiting.")
            return False
        
        target_row, target_col = cell["row"], cell["col"]
        print(f"Target located at: ({target_row}, {target_col})")

        # Step 3: Navigate to Target
        print("Navigating to target...")
        success = self.navigate_to_cell(target_row, target_col, label)

        print(f"\n{'='*60}")
        if success:
            print(f"Goal complete: Found '{label}'")
        else:
            print(f"Goal incomplete: Could not confirm '{label}'")
        print(f"{'='*60}\n")

        return success