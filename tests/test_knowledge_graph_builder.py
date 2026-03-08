from mantra.knowledge_graph_builder import KnowledgeGraphBuilder


def test_knowledge_graph_builder_builds_nodes_and_edges():
    builder = KnowledgeGraphBuilder()
    graph = builder.build_from_records(
        [
            {"user_id": "u1", "track_id": "t1", "artist": "a1", "genre": "ambient", "playlist_id": "p1"},
            {"user_id": "u2", "track_id": "t2", "artist": "a2", "genre": "house", "playlist_id": "p2"},
        ]
    )
    assert "u1" in graph["nodes"]
    assert len(graph["edges"]) >= 4

