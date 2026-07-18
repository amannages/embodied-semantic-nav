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
        Given a semantic label, find the best known position in the semantic map.
        Returns (x, z) coordinates in world space, or None if not found.
        """
        best_location = self.semantic_map.best_known_location(label)
        if best_location is None:
            print(f"No known location for label '{label}' in the semantic map.")
            return None
        return best_location["position"]
    