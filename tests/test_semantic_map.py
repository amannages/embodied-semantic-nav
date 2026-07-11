from mapping.semantic_map import SemanticMap


def test_update_stores_detection_in_grid_cell():
    sem_map = SemanticMap(width=20, height=20, resolution=1.0)

    sem_map.update([
        {"label": "Microwave", "confidence": 0.87}
    ], agent_x=1.0, agent_z=2.0)

    row, col = sem_map.world_to_grid(1.0, 2.0)
    assert sem_map.get_cell_confidence("Microwave", row, col) == 0.87
    assert sem_map.has_seen("Microwave") is True
