import pytest
import json
from pathlib import Path
from graph_structure import TokenGraph, Node


@pytest.fixture
def small_graph() -> TokenGraph:
    """5-node connected cycle for unit tests. No JSON file required.

    Cycle topology (bidirectional):
      cube0_cluster0 <-> cube0_cluster1 <-> cube1_cluster0
           ^                                      |
           |                                      v
      cube2_cluster0 <-> cube1_cluster1 <---------+

    Notable properties:
    - "royal" appears in BOTH cube0_cluster0 and cube0_cluster1 (multi-node token).
    - cube2_cluster0 tokens ("apple","banana","cherry") share nothing with other nodes.
    - All nodes are reachable from all other nodes (1 connected component).
    """
    graph = TokenGraph()

    node_specs = [
        (
            "cube0_cluster0",
            [1, 2, 3],
            ["king", "queen", "royal"],
            {"x": 0.1, "y": 0.2},
            {"std_x": 0.01, "std_y": 0.01},
            {"cube0_cluster1", "cube2_cluster0"},
        ),
        (
            "cube0_cluster1",
            [4, 5, 6],
            ["royal", "throne", "crown"],
            {"x": 0.2, "y": 0.3},
            {"std_x": 0.01, "std_y": 0.01},
            {"cube0_cluster0", "cube1_cluster0"},
        ),
        (
            "cube1_cluster0",
            [7, 8, 9],
            ["man", "woman", "person"],
            {"x": 0.5, "y": 0.5},
            {"std_x": 0.02, "std_y": 0.02},
            {"cube0_cluster1", "cube1_cluster1"},
        ),
        (
            "cube1_cluster1",
            [10, 11, 12],
            ["doctor", "nurse", "teacher"],
            {"x": 0.7, "y": 0.4},
            {"std_x": 0.01, "std_y": 0.01},
            {"cube1_cluster0", "cube2_cluster0"},
        ),
        (
            "cube2_cluster0",
            [13, 14, 15],
            ["apple", "banana", "cherry"],
            {"x": 0.3, "y": 0.1},
            {"std_x": 0.01, "std_y": 0.01},
            {"cube1_cluster1", "cube0_cluster0"},
        ),
    ]

    for node_id, token_ids, tokens, position, spread, connected_nodes in node_specs:
        graph.nodes[node_id] = Node(
            id=node_id,
            token_ids=token_ids,
            tokens=tokens,
            size=len(tokens),
            position=position,
            spread=spread,
            connected_nodes=connected_nodes,
        )

    # weighted_bfs_path requires self.edge_weights — not set in TokenGraph.__init__
    graph.edge_weights = {
        ("cube0_cluster0", "cube0_cluster1"): 0.9,
        ("cube0_cluster1", "cube1_cluster0"): 0.7,
        ("cube1_cluster0", "cube1_cluster1"): 0.8,
        ("cube1_cluster1", "cube2_cluster0"): 0.6,
        ("cube2_cluster0", "cube0_cluster0"): 0.5,
    }

    return graph


@pytest.fixture
def make_graph():
    """Factory fixture: builds a TokenGraph from a {node_id: (tokens, connections)} dict.

    Usage:
        graph = make_graph({
            "a": (["hello", "world"], ["b"]),
            "b": (["foo"], ["a"]),
        })
    """
    def _make(spec: dict) -> TokenGraph:
        graph = TokenGraph()
        graph.edge_weights = {}
        for i, (node_id, (tokens, connections)) in enumerate(spec.items()):
            graph.nodes[node_id] = Node(
                id=node_id,
                token_ids=list(range(i * 10, i * 10 + len(tokens))),
                tokens=tokens,
                size=len(tokens),
                position={"x": float(i), "y": 0.0},
                spread={"std_x": 0.0, "std_y": 0.0},
                connected_nodes=set(connections),
            )
        return graph

    return _make


@pytest.fixture(scope="session")
def full_graph() -> TokenGraph:
    """Full production graph loaded once per session.

    Requires node_clusters_with_weights.json in the working directory.
    Only used by @pytest.mark.slow tests.
    """
    graph = TokenGraph.from_json("node_clusters_with_weights.json")
    if not hasattr(graph, "edge_weights"):
        graph.edge_weights = {}
    return graph


@pytest.fixture
def small_graph_json(tmp_path, small_graph) -> Path:
    """Writes small_graph to a temporary JSON file.

    Used by from_json tests and CLI tests to avoid coupling to real data files.
    """
    data = {
        "nodes": {
            node_id: {
                "token_ids": node.token_ids,
                "tokens": node.tokens,
                "size": node.size,
                "position": node.position,
                "cluster_spread": node.spread,
                "connected_nodes": list(node.connected_nodes),
            }
            for node_id, node in small_graph.nodes.items()
        }
    }
    path = tmp_path / "test_graph.json"
    path.write_text(json.dumps(data))
    return path
