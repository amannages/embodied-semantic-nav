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
        )).keys()

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
        
        """
        
    