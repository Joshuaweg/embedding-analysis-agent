"""PydanticAI-based topology analysis agent.

Connects to a local Ollama instance via the OpenAI-compatible endpoint.
All graph tools are registered as module-level functions from agent_tools.py,
avoiding the class-method / @tool decorator issue that broke the previous Agno version.
"""
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel
from pydantic_ai.providers.openai import OpenAIProvider

from agent_tools import (
    Deps,
    analyze_components,
    analyze_network,
    analyze_paths,
    analyze_token_patterns,
    analyze_value_neighborhood,
    bfs_path,
    compare_value_poles,
    compute_graph_statistics,
    compute_node_centrality,
    detect_communities,
    extract_subgraph,
    find_all_paths,
    find_nodes_with_token,
    find_value_bridges,
    get_connected_nodes,
    get_hypercube_nodes,
    get_node_info,
    random_walk,
    weighted_bfs_path,
)
from system_prompts import TOPOLOGY_AGENT_PROMPT


def create_agent(model_name: str = "llama3.1:8b") -> Agent:
    """Create and return a configured topology analysis agent.

    Args:
        model_name: Ollama model name to use (default: llama3.1:8b).
                    Any model available in your local Ollama installation works.
                    Recommended models with good tool calling: llama3.1, qwen2.5, deepseek-r1.

    Returns:
        A PydanticAI Agent with all 19 graph tools registered.
    """
    model = OpenAIModel(
        model_name,
        provider=OpenAIProvider(
            base_url="http://localhost:11434/v1",
            api_key="ollama",  # Required field, ignored by Ollama
        ),
    )

    agent: Agent[Deps, str] = Agent(
        model,
        deps_type=Deps,
        output_type=str,
        system_prompt=TOPOLOGY_AGENT_PROMPT,
        retries=3,
    )

    # Register all 16 tools
    agent.tool(get_node_info)
    agent.tool(get_connected_nodes)
    agent.tool(get_hypercube_nodes)
    agent.tool(find_nodes_with_token)
    agent.tool(bfs_path)
    agent.tool(find_all_paths)
    agent.tool(analyze_components)
    agent.tool(random_walk)
    agent.tool(analyze_network)
    agent.tool(detect_communities)
    agent.tool(compute_node_centrality)
    agent.tool(extract_subgraph)
    agent.tool(analyze_token_patterns)
    agent.tool(analyze_paths)
    agent.tool(compute_graph_statistics)
    agent.tool(weighted_bfs_path)

    # Value system analysis tools
    agent.tool(analyze_value_neighborhood)
    agent.tool(compare_value_poles)
    agent.tool(find_value_bridges)

    return agent
