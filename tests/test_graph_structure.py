"""Tests for graph_structure.py — TokenGraph and Node behavior.

All tests use in-memory fixtures (small_graph, make_graph) unless marked @slow.
No LLM or agent calls are made here.
"""
import pytest
from graph_structure import TokenGraph, Node


# ---------------------------------------------------------------------------
# TokenGraph.from_json
# ---------------------------------------------------------------------------

def test_from_json_loads_all_nodes(small_graph_json):
    graph = TokenGraph.from_json(str(small_graph_json))
    assert len(graph.nodes) == 5


def test_from_json_node_fields_have_correct_types(small_graph_json):
    graph = TokenGraph.from_json(str(small_graph_json))
    node = graph.nodes["cube0_cluster0"]
    assert node.id == "cube0_cluster0"
    assert isinstance(node.token_ids, list)
    assert isinstance(node.tokens, list)
    assert isinstance(node.size, int)
    assert isinstance(node.position, dict)
    assert "x" in node.position and "y" in node.position
    assert isinstance(node.connected_nodes, set)


def test_from_json_missing_file_raises_file_not_found():
    with pytest.raises(FileNotFoundError):
        TokenGraph.from_json("definitely_does_not_exist_xyz.json")


# ---------------------------------------------------------------------------
# TokenGraph.get_node
# ---------------------------------------------------------------------------

def test_get_node_returns_node_with_correct_id(small_graph):
    node = small_graph.get_node("cube0_cluster0")
    assert node is not None
    assert node.id == "cube0_cluster0"


def test_get_node_missing_id_returns_none(small_graph):
    node = small_graph.get_node("cube99_cluster99")
    assert node is None


# ---------------------------------------------------------------------------
# TokenGraph.find_nodes_with_token
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("token,expected_count", [
    ("king",  1),   # present in exactly one node
    ("royal", 2),   # present in cube0_cluster0 AND cube0_cluster1
    ("apple", 1),   # present in exactly one node
    ("zzz",   0),   # not in any node
    ("",      0),   # empty string — must not match anything
], ids=["single-node", "multi-node", "leaf-token", "absent", "empty-string"])
def test_find_nodes_with_token(small_graph, token, expected_count):
    result = small_graph.find_nodes_with_token(token)
    assert len(result) == expected_count
    # Every returned node must actually contain the token
    assert all(token in node.tokens for node in result)


# ---------------------------------------------------------------------------
# TokenGraph.bfs_path
# ---------------------------------------------------------------------------

def test_bfs_path_adjacent_nodes_returns_length_2(small_graph):
    path = small_graph.bfs_path("cube0_cluster0", "cube0_cluster1")
    assert len(path) == 2
    assert path[0].id == "cube0_cluster0"
    assert path[-1].id == "cube0_cluster1"


def test_bfs_path_multihop_returns_shortest_path(small_graph):
    # Shortest path: cube0_cluster0 → cube0_cluster1 → cube1_cluster0 = 3 nodes
    # Longer alternative: cube0_cluster0 → cube2_cluster0 → cube1_cluster1 → cube1_cluster0 = 4 nodes
    # BFS must find the 3-node path.
    path = small_graph.bfs_path("cube0_cluster0", "cube1_cluster0")
    assert len(path) == 3
    assert path[0].id == "cube0_cluster0"
    assert path[-1].id == "cube1_cluster0"
    # BFS guarantees no repeated nodes
    ids = [n.id for n in path]
    assert len(ids) == len(set(ids)), "BFS path should not contain repeated nodes"


def test_bfs_path_same_node_returns_length_1(small_graph):
    path = small_graph.bfs_path("cube0_cluster0", "cube0_cluster0")
    assert len(path) == 1
    assert path[0].id == "cube0_cluster0"


def test_bfs_path_disconnected_returns_empty(make_graph):
    graph = make_graph({
        "a": (["hello"], []),
        "b": (["world"], []),
    })
    result = graph.bfs_path("a", "b")
    assert result == []


def test_bfs_path_invalid_start_node_returns_empty(small_graph):
    result = small_graph.bfs_path("cube99_cluster99", "cube0_cluster0")
    assert result == []


def test_bfs_path_invalid_end_node_returns_empty(small_graph):
    result = small_graph.bfs_path("cube0_cluster0", "cube99_cluster99")
    assert result == []


# ---------------------------------------------------------------------------
# TokenGraph.analyze_components
# ---------------------------------------------------------------------------

def test_analyze_components_returns_required_keys(small_graph):
    result = small_graph.analyze_components()
    for key in ["total_components", "component_sizes", "node_components",
                "largest_component", "components"]:
        assert key in result, f"Missing key: '{key}'"


def test_analyze_components_single_connected_graph(small_graph):
    result = small_graph.analyze_components()
    assert result["total_components"] == 1


def test_analyze_components_sizes_sum_to_node_count(small_graph):
    result = small_graph.analyze_components()
    total = sum(size for _, size, _ in result["component_sizes"])
    assert total == len(small_graph.nodes)


def test_analyze_components_two_isolated_nodes(make_graph):
    graph = make_graph({
        "a": (["hello"], []),
        "b": (["world"], []),
    })
    result = graph.analyze_components()
    assert result["total_components"] == 2
    total = sum(size for _, size, _ in result["component_sizes"])
    assert total == 2


# ---------------------------------------------------------------------------
# TokenGraph.random_walk
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("num_steps", [1, 3, 5],
                         ids=["1-step", "3-step", "5-step"])
def test_random_walk_length_equals_steps_plus_one(small_graph, num_steps):
    # All nodes in small_graph have neighbors, so no early termination
    walk = small_graph.random_walk("cube0_cluster0", num_steps)
    assert len(walk) == num_steps + 1
    assert walk[0].id == "cube0_cluster0"


def test_random_walk_all_nodes_in_graph(small_graph):
    walk = small_graph.random_walk("cube0_cluster0", 5)
    for node in walk:
        assert node.id in small_graph.nodes


def test_random_walk_invalid_start_raises_value_error(small_graph):
    with pytest.raises(ValueError, match="cube99_cluster99"):
        small_graph.random_walk("cube99_cluster99", 3)


def test_random_walk_no_start_selects_randomly(small_graph):
    walk = small_graph.random_walk(num_steps=3)
    assert len(walk) == 4
    assert walk[0].id in small_graph.nodes


# ---------------------------------------------------------------------------
# TokenGraph.find_all_paths
# ---------------------------------------------------------------------------

def test_find_all_paths_respects_depth_limit(small_graph):
    max_depth = 3
    paths = small_graph.find_all_paths("cube0_cluster0", "cube1_cluster1", max_depth)
    for path in paths:
        # find_all_paths returns when len(path) > max_depth, so max length = max_depth
        assert len(path) <= max_depth, (
            f"Path {[n.id for n in path]} has length {len(path)} > max_depth {max_depth}"
        )


def test_find_all_paths_endpoints_correct(small_graph):
    paths = small_graph.find_all_paths("cube0_cluster0", "cube1_cluster0", max_depth=5)
    assert len(paths) >= 1
    for path in paths:
        assert path[0].id == "cube0_cluster0"
        assert path[-1].id == "cube1_cluster0"


def test_find_all_paths_disconnected_returns_empty(make_graph):
    graph = make_graph({
        "a": (["hello"], []),
        "b": (["world"], []),
    })
    result = graph.find_all_paths("a", "b", max_depth=5)
    assert result == []


# ---------------------------------------------------------------------------
# TokenGraph.compute_graph_statistics
# Note: requires a connected graph — small_graph is a cycle so this is safe.
# ---------------------------------------------------------------------------

def test_compute_graph_statistics_returns_required_keys(small_graph):
    result = small_graph.compute_graph_statistics()
    for key in ["avg_degree", "density", "clustering_coefficient",
                "diameter", "avg_shortest_path"]:
        assert key in result, f"Missing key: '{key}'"


def test_compute_graph_statistics_values_in_valid_range(small_graph):
    result = small_graph.compute_graph_statistics()
    assert result["avg_degree"] > 0
    assert 0.0 <= result["density"] <= 1.0
    assert result["diameter"] >= 1


# ---------------------------------------------------------------------------
# TokenGraph.weighted_bfs_path
# Note: returns list of node ID strings (not Node objects), unlike bfs_path.
# ---------------------------------------------------------------------------

def test_weighted_bfs_path_returns_list_of_strings(small_graph):
    result = small_graph.weighted_bfs_path("cube0_cluster0", "cube1_cluster0")
    assert isinstance(result, list)
    assert all(isinstance(item, str) for item in result)


def test_weighted_bfs_path_correct_endpoints(small_graph):
    result = small_graph.weighted_bfs_path("cube0_cluster0", "cube1_cluster0")
    assert len(result) >= 2
    assert result[0] == "cube0_cluster0"
    assert result[-1] == "cube1_cluster0"


def test_weighted_bfs_path_invalid_node_returns_empty(small_graph):
    result = small_graph.weighted_bfs_path("cube99_cluster99", "cube0_cluster0")
    assert result == []


# ---------------------------------------------------------------------------
# Slow tests — use full production graph
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_full_graph_node_count(full_graph):
    assert len(full_graph.nodes) > 1000


@pytest.mark.slow
def test_full_graph_bfs_finds_path(full_graph):
    node_ids = list(full_graph.nodes.keys())
    start, end = node_ids[0], node_ids[100]
    path = full_graph.bfs_path(start, end)
    # Path may be empty if disconnected, but should not raise
    assert isinstance(path, list)
