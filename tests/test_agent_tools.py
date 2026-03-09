"""Tests for agent_tools.py — PydanticAI tool functions.

Tools are tested by calling them directly with a minimal mock context,
avoiding any LLM calls. This validates logic independently of the agent loop.

All 16 tools are verified to return str via the parametrized test at the bottom.
"""
import importlib
import pytest
from unittest.mock import MagicMock


def make_ctx(graph):
    """Build a minimal RunContext-compatible object for testing tools directly.

    Tools only access ctx.deps.graph, so a MagicMock with Deps injected is sufficient.
    """
    from agent_tools import Deps
    ctx = MagicMock()
    ctx.deps = Deps(graph=graph)
    return ctx


# ---------------------------------------------------------------------------
# get_node_info
# ---------------------------------------------------------------------------

def test_get_node_info_returns_non_empty_str(small_graph):
    from agent_tools import get_node_info
    ctx = make_ctx(small_graph)
    result = get_node_info(ctx, "cube0_cluster0")
    assert isinstance(result, str)
    assert len(result) > 0


def test_get_node_info_contains_node_id(small_graph):
    from agent_tools import get_node_info
    ctx = make_ctx(small_graph)
    result = get_node_info(ctx, "cube0_cluster0")
    assert "cube0_cluster0" in result


def test_get_node_info_bad_id_raises_model_retry(small_graph):
    from agent_tools import get_node_info
    from pydantic_ai import ModelRetry
    ctx = make_ctx(small_graph)
    with pytest.raises(ModelRetry):
        get_node_info(ctx, "cube99_cluster99")


# ---------------------------------------------------------------------------
# get_connected_nodes
# ---------------------------------------------------------------------------

def test_get_connected_nodes_returns_str(small_graph):
    from agent_tools import get_connected_nodes
    ctx = make_ctx(small_graph)
    result = get_connected_nodes(ctx, "cube0_cluster0")
    assert isinstance(result, str)


def test_get_connected_nodes_contains_neighbor_ids(small_graph):
    from agent_tools import get_connected_nodes
    ctx = make_ctx(small_graph)
    result = get_connected_nodes(ctx, "cube0_cluster0")
    # cube0_cluster0 is connected to cube0_cluster1 and cube2_cluster0
    assert "cube0_cluster1" in result or "cube2_cluster0" in result


def test_get_connected_nodes_bad_id_raises_model_retry(small_graph):
    from agent_tools import get_connected_nodes
    from pydantic_ai import ModelRetry
    ctx = make_ctx(small_graph)
    with pytest.raises(ModelRetry):
        get_connected_nodes(ctx, "cube99_cluster99")


# ---------------------------------------------------------------------------
# find_nodes_with_token
# ---------------------------------------------------------------------------

def test_find_nodes_with_token_tool_returns_str(small_graph):
    from agent_tools import find_nodes_with_token
    ctx = make_ctx(small_graph)
    result = find_nodes_with_token(ctx, "king")
    assert isinstance(result, str)


def test_find_nodes_with_token_tool_absent_token_returns_str(small_graph):
    from agent_tools import find_nodes_with_token
    ctx = make_ctx(small_graph)
    result = find_nodes_with_token(ctx, "zzz_not_a_token")
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# bfs_path
# ---------------------------------------------------------------------------

def test_bfs_path_tool_returns_str(small_graph):
    from agent_tools import bfs_path
    ctx = make_ctx(small_graph)
    result = bfs_path(ctx, "cube0_cluster0", "cube0_cluster1")
    assert isinstance(result, str)


def test_bfs_path_tool_disconnected_returns_no_path_message(make_graph):
    from agent_tools import bfs_path
    graph = make_graph({
        "a": (["hello"], []),
        "b": (["world"], []),
    })
    ctx = make_ctx(graph)
    result = bfs_path(ctx, "a", "b")
    assert isinstance(result, str)
    # Must clearly communicate that no path was found
    lower = result.lower()
    assert "no path" in lower or "not found" in lower or "disconnected" in lower, (
        f"Expected 'no path' message, got: {result!r}"
    )


# ---------------------------------------------------------------------------
# analyze_components
# ---------------------------------------------------------------------------

def test_analyze_components_tool_returns_non_empty_str(small_graph):
    from agent_tools import analyze_components
    ctx = make_ctx(small_graph)
    result = analyze_components(ctx)
    assert isinstance(result, str)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# random_walk
# ---------------------------------------------------------------------------

def test_random_walk_tool_returns_str(small_graph):
    from agent_tools import random_walk
    ctx = make_ctx(small_graph)
    result = random_walk(ctx, "cube0_cluster0", 3)
    assert isinstance(result, str)


# ---------------------------------------------------------------------------
# All 16 tools must return str — parametrized exhaustive check
# ---------------------------------------------------------------------------

TOOL_CALLS = [
    ("get_node_info",           ["cube0_cluster0"]),
    ("get_connected_nodes",     ["cube0_cluster0"]),
    ("get_hypercube_nodes",     [0]),
    ("find_nodes_with_token",   ["king"]),
    ("bfs_path",                ["cube0_cluster0", "cube0_cluster1"]),
    ("find_all_paths",          ["cube0_cluster0", "cube1_cluster0", 3]),
    ("analyze_components",      []),
    ("random_walk",             ["cube0_cluster0", 3]),
    ("analyze_network",         []),
    ("detect_communities",      []),
    ("compute_node_centrality", []),
    ("extract_subgraph",        [["cube0_cluster0", "cube0_cluster1"]]),
    ("analyze_token_patterns",  []),
    ("analyze_paths",           ["cube0_cluster0", 3]),
    ("compute_graph_statistics", []),
    ("weighted_bfs_path",       ["cube0_cluster0", "cube1_cluster0"]),
]


@pytest.mark.parametrize("tool_name,args", TOOL_CALLS, ids=[t[0] for t in TOOL_CALLS])
def test_all_tools_return_str(small_graph, tool_name, args):
    tools_module = importlib.import_module("agent_tools")
    tool_fn = getattr(tools_module, tool_name)
    ctx = make_ctx(small_graph)
    result = tool_fn(ctx, *args)
    assert isinstance(result, str), (
        f"Tool '{tool_name}' returned {type(result).__name__}, expected str"
    )
    assert len(result) > 0, f"Tool '{tool_name}' returned empty string"
