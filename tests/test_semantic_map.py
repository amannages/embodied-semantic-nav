from mapping.semantic_map import SemanticMap


def test_update_stores_detection_in_grid_cell():
    sem_map = SemanticMap(width=20, height=20, resolution=1.0)

    sem_map.update([
        {"label": "Microwave", "confidence": 0.87}
    ], agent_x=1.0, agent_z=2.0)

    row, col = sem_map.world_to_grid(1.0, 2.0)
    assert sem_map.get_cell_confidence("Microwave", row, col) == 0.87
    assert sem_map.has_seen("Microwave") is True


def test_semantic_map_queries_cells_by_label():
    sem_map = SemanticMap(width=20, height=20, resolution=1.0)

    sem_map.update([{"label": "Mug", "confidence": 0.62}], agent_x=0.0, agent_z=0.0)
    sem_map.update([{"label": "Mug", "confidence": 0.91}], agent_x=2.0, agent_z=1.0)

    nearest = sem_map.nearest_cell_with_label("Mug", agent_x=0.1, agent_z=0.1)
    assert nearest is not None
    assert nearest["position"] == (0.0, 0.0)

    best = sem_map.best_known_location("Mug")
    assert best is not None
    assert best["position"] == (2.0, 1.0)
    assert best["confidence"] == 0.91
