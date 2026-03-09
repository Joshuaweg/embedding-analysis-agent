"""Tests for topology_agent.py — PydanticAI Agent structure.

No real LLM calls are made. TestModel is used for run tests.
These tests verify that the agent is correctly assembled:
  - create_agent() returns a PydanticAI Agent
  - All 16 tools are registered
  - Deps type is correctly set
  - Agent completes a run with TestModel
"""
import pytest
from pydantic_ai import Agent
from pydantic_ai.models.test import TestModel


EXPECTED_TOOLS = {
    "get_node_info",
    "get_connected_nodes",
    "get_hypercube_nodes",
    "find_nodes_with_token",
    "bfs_path",
    "find_all_paths",
    "analyze_components",
    "random_walk",
    "analyze_network",
    "detect_communities",
    "compute_node_centrality",
    "extract_subgraph",
    "analyze_token_patterns",
    "analyze_paths",
    "compute_graph_statistics",
    "weighted_bfs_path",
}


def test_create_agent_returns_pydantic_ai_agent():
    from topology_agent import create_agent
    agent = create_agent()
    assert isinstance(agent, Agent)


def test_agent_has_all_16_tools_registered():
    from topology_agent import create_agent
    agent = create_agent()
    registered = set(agent._function_toolset.tools.keys())
    missing = EXPECTED_TOOLS - registered
    extra = registered - EXPECTED_TOOLS
    assert not missing, f"Tools not registered: {missing}"
    assert not extra, f"Unexpected tools registered: {extra}"


def test_agent_deps_type_is_deps():
    from topology_agent import create_agent
    from agent_tools import Deps
    agent = create_agent()
    assert agent._deps_type is Deps


def test_agent_runs_with_test_model(small_graph):
    from topology_agent import create_agent
    from agent_tools import Deps
    agent = create_agent()
    # call_tools=[] prevents TestModel from invoking tools with synthetic (invalid) args
    with agent.override(model=TestModel(call_tools=[])):
        result = agent.run_sync("What nodes are in the graph?", deps=Deps(graph=small_graph))
    assert result is not None
    assert isinstance(result.output, str)


def test_agent_accepts_custom_model_name():
    from topology_agent import create_agent
    # Should not raise — model name is passed through to OpenAIModel
    agent = create_agent(model_name="llama3.2:3b")
    assert isinstance(agent, Agent)
