"""PydanticAI tool functions for the topology analysis agent.

All tools are module-level functions registered on the agent via @agent.tool.
Shared state (the TokenGraph) is injected via RunContext[Deps], eliminating
the class-method pattern that broke the previous Agno implementation.

Each tool returns a str so PydanticAI can pass the result back to the LLM.
ModelRetry is raised when the LLM provides a bad node ID, prompting it to
self-correct (e.g., by first calling find_nodes_with_token).
"""
import json
from dataclasses import dataclass
from typing import Annotated, List

from pydantic import Field
from pydantic_ai import ModelRetry, RunContext

from graph_structure import TokenGraph


@dataclass
class Deps:
    """Shared dependencies injected into every tool call."""
    graph: TokenGraph


# ---------------------------------------------------------------------------
# Node lookup
# ---------------------------------------------------------------------------

def get_node_info(
    ctx: RunContext[Deps],
    node_id: Annotated[str, Field(description="Node ID in the form 'cubeX_clusterY', e.g. 'cube0_cluster0'")],
) -> str:
    """Get detailed information about a specific node, including its tokens and connections."""
    node = ctx.deps.graph.get_node(node_id)
    if node is None:
        raise ModelRetry(
            f"Node '{node_id}' not found. "
            "Use find_nodes_with_token to locate a node by its token content first."
        )
    return ctx.deps.graph.get_node_info(node_id)


def get_connected_nodes(
    ctx: RunContext[Deps],
    node_id: Annotated[str, Field(description="Node ID to find neighbours for")],
) -> str:
    """Get all nodes directly connected to a given node."""
    node = ctx.deps.graph.get_node(node_id)
    if node is None:
        raise ModelRetry(
            f"Node '{node_id}' not found. "
            "Use find_nodes_with_token to locate a node by its token content first."
        )
    connected = ctx.deps.graph.get_connected_nodes(node_id)
    return json.dumps([n.id for n in connected])


def get_hypercube_nodes(
    ctx: RunContext[Deps],
    cube_num: Annotated[int, Field(description="Hypercube number (the X in cubeX_clusterY)", ge=0)],
) -> str:
    """Get all cluster nodes within a specific hypercube."""
    nodes = ctx.deps.graph.get_hypercube_nodes(cube_num)
    if not nodes:
        return f"No nodes found in hypercube {cube_num}."
    return json.dumps([n.id for n in nodes])


# ---------------------------------------------------------------------------
# Token search
# ---------------------------------------------------------------------------

def find_nodes_with_token(
    ctx: RunContext[Deps],
    token: Annotated[str, Field(description="Token string to search for, e.g. 'king'")],
) -> str:
    """Find all graph nodes that contain a specific token.

    Use this first to locate relevant nodes before calling path-finding tools.
    """
    nodes = ctx.deps.graph.find_nodes_with_token(token)
    if not nodes:
        return f"No nodes found containing the token '{token}'."
    result = {
        "token": token,
        "count": len(nodes),
        "node_ids": [n.id for n in nodes],
    }
    return json.dumps(result)


# ---------------------------------------------------------------------------
# Path finding
# ---------------------------------------------------------------------------

def bfs_path(
    ctx: RunContext[Deps],
    start_node_id: Annotated[str, Field(description="Starting node ID, e.g. 'cube0_cluster0'")],
    end_node_id: Annotated[str, Field(description="Target node ID")],
) -> str:
    """Find the shortest path between two nodes using breadth-first search.

    Returns the sequence of node IDs from start to end.
    If the nodes are in disconnected components, reports that no path exists.
    """
    path = ctx.deps.graph.bfs_path(start_node_id, end_node_id)
    if not path:
        return (
            f"No path found between '{start_node_id}' and '{end_node_id}'. "
            "They may be in disconnected components."
        )
    node_ids = [n.id for n in path]
    tokens_along_path = [n.tokens[:3] for n in path]
    return json.dumps({"path": node_ids, "length": len(node_ids), "sample_tokens": tokens_along_path})


def find_all_paths(
    ctx: RunContext[Deps],
    start_node_id: Annotated[str, Field(description="Starting node ID")],
    end_node_id: Annotated[str, Field(description="Target node ID")],
    max_depth: Annotated[int, Field(description="Maximum path length to search (1–10)", ge=1, le=10)] = 5,
) -> str:
    """Find all paths between two nodes up to a maximum depth.

    Returns all discovered paths. Use a small max_depth (3–5) for large graphs.
    """
    paths = ctx.deps.graph.find_all_paths(start_node_id, end_node_id, max_depth)
    if not paths:
        return f"No paths found between '{start_node_id}' and '{end_node_id}' within depth {max_depth}."
    result = [[n.id for n in path] for path in paths]
    return json.dumps({"count": len(result), "paths": result})


def weighted_bfs_path(
    ctx: RunContext[Deps],
    start_node_id: Annotated[str, Field(description="Starting node ID")],
    end_node_id: Annotated[str, Field(description="Target node ID")],
) -> str:
    """Find the path that maximises edge weights between two nodes.

    Prefer this over bfs_path when you want the most semantically similar path
    rather than the shortest hop count.
    """
    try:
        path = ctx.deps.graph.weighted_bfs_path(start_node_id, end_node_id)
    except (AttributeError, TypeError):
        return (
            "Weighted path finding is unavailable — edge weights are missing or invalid for this graph. "
            "Use bfs_path instead."
        )
    if not path:
        return (
            f"No weighted path found between '{start_node_id}' and '{end_node_id}'."
        )
    return json.dumps({"path": path, "length": len(path)})


# ---------------------------------------------------------------------------
# Graph traversal
# ---------------------------------------------------------------------------

def random_walk(
    ctx: RunContext[Deps],
    start_node_id: Annotated[str, Field(description="Starting node ID for the walk")],
    num_steps: Annotated[int, Field(description="Number of steps to take (1–50)", ge=1, le=50)] = 5,
) -> str:
    """Perform a random walk through the graph starting from a given node.

    Useful for exploring the local neighbourhood of a token cluster.
    """
    try:
        walk = ctx.deps.graph.random_walk(start_node_id, num_steps)
    except ValueError as e:
        raise ModelRetry(str(e))
    result = [{"node": n.id, "sample_tokens": n.tokens[:3]} for n in walk]
    return json.dumps(result)


def analyze_paths(
    ctx: RunContext[Deps],
    start_node_id: Annotated[str, Field(description="Node ID to explore paths from")],
    max_length: Annotated[int, Field(description="Maximum path length (1–10)", ge=1, le=10)] = 5,
) -> str:
    """Analyse all paths radiating outward from a node up to a maximum length.

    Returns paths as lists of node IDs. Useful for mapping a node's reachable neighbourhood.
    """
    node = ctx.deps.graph.get_node(start_node_id)
    if node is None:
        raise ModelRetry(
            f"Node '{start_node_id}' not found. "
            "Use find_nodes_with_token to locate a node first."
        )
    paths = ctx.deps.graph.analyze_paths(start_node_id, max_length)
    return json.dumps({"start": start_node_id, "path_count": len(paths), "paths": paths})


# ---------------------------------------------------------------------------
# Component and community analysis
# ---------------------------------------------------------------------------

def analyze_components(ctx: RunContext[Deps]) -> str:
    """Analyse the connected components of the graph.

    Returns the number of components, their sizes, and which nodes belong to each.
    """
    result = ctx.deps.graph.analyze_components()
    summary = {
        "total_components": result["total_components"],
        "largest_component_size": result["largest_component"][1] if result["largest_component"] else 0,
        "component_sizes": [(comp_id, size) for comp_id, size, _ in result["component_sizes"]],
    }
    return json.dumps(summary)


def detect_communities(ctx: RunContext[Deps]) -> str:
    """Detect communities in the graph using the Louvain method.

    Returns the number of communities, modularity score, and community sizes.
    High modularity (>0.3) indicates well-separated communities.
    """
    result = ctx.deps.graph.detect_communities()
    summary = {
        "num_communities": len(result["communities"]),
        "modularity": round(result["modularity"], 4),
        "community_sizes": sorted([len(c) for c in result["communities"]], reverse=True),
    }
    return json.dumps(summary)


def extract_subgraph(
    ctx: RunContext[Deps],
    node_ids: Annotated[List[str], Field(description="List of node IDs to include in the subgraph")],
) -> str:
    """Extract a subgraph containing specified nodes and all their immediate neighbours.

    Useful for zooming in on a region of interest.
    """
    missing = [nid for nid in node_ids if ctx.deps.graph.get_node(nid) is None]
    if missing:
        raise ModelRetry(
            f"Node IDs not found: {missing}. "
            "Use find_nodes_with_token to locate valid node IDs first."
        )
    subgraph = ctx.deps.graph.extract_subgraph(node_ids)
    result = {
        "node_count": len(subgraph),
        "nodes": {nid: {"tokens": node.tokens[:5], "size": node.size} for nid, node in subgraph.items()},
    }
    return json.dumps(result)


# ---------------------------------------------------------------------------
# Network and centrality metrics
# ---------------------------------------------------------------------------

def analyze_network(ctx: RunContext[Deps]) -> str:
    """Perform social network analysis: density, clustering, connected components, diameter.

    Returns global graph metrics. Note: may be slow on large graphs.
    """
    result = ctx.deps.graph.analyze_network()
    global_m = result["global_metrics"]
    summary = {
        "num_nodes": global_m["num_nodes"],
        "num_edges": global_m["num_edges"],
        "density": round(global_m["density"], 4),
        "average_clustering": round(global_m["average_clustering"], 4),
        "is_connected": global_m["is_connected"],
        "num_components": global_m["number_connected_components"],
    }
    return json.dumps(summary)


def compute_node_centrality(ctx: RunContext[Deps]) -> str:
    """Compute degree, betweenness, eigenvector, and closeness centrality for all nodes.

    Returns the top 10 most central nodes by each metric. Central nodes are
    likely to be semantically influential in the embedding space.
    """
    result = ctx.deps.graph.compute_node_centrality()
    top_n = 10

    def top(metric_dict: dict) -> list:
        return sorted(metric_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]

    summary = {
        "top_degree":      [{"node": n, "score": round(s, 4)} for n, s in top(result["degree"])],
        "top_betweenness": [{"node": n, "score": round(s, 4)} for n, s in top(result["betweenness"])],
        "top_closeness":   [{"node": n, "score": round(s, 4)} for n, s in top(result["closeness"])],
        "top_eigenvector": [{"node": n, "score": round(s, 4)} for n, s in top(result.get("eigenvector", {}))],
    }
    return json.dumps(summary)


def compute_graph_statistics(ctx: RunContext[Deps]) -> str:
    """Compute graph-level statistics: average degree, density, clustering coefficient, diameter.

    Note: requires a connected graph. Call analyze_components first if unsure.
    """
    try:
        result = ctx.deps.graph.compute_graph_statistics()
    except Exception as e:
        return json.dumps({"error": str(e), "hint": "Graph may be disconnected. Try analyze_components first."})
    return json.dumps({k: round(v, 4) if isinstance(v, float) else v for k, v in result.items()})


# ---------------------------------------------------------------------------
# Token pattern analysis
# ---------------------------------------------------------------------------

def analyze_token_patterns(ctx: RunContext[Deps]) -> str:
    """Analyse token frequency and co-occurrence patterns across all nodes.

    Returns the 20 most frequent tokens and 20 most common token pairs.
    Useful for identifying dominant concepts and potential biases.
    """
    result = ctx.deps.graph.analyze_token_patterns()
    top_tokens = sorted(result["token_frequencies"].items(), key=lambda x: x[1], reverse=True)[:20]
    summary = {
        "top_tokens": [{"token": t, "count": c} for t, c in top_tokens],
        "common_pairs": [
            {"pair": list(pair), "count": count}
            for pair, count in result["common_pairs"]
        ],
    }
    return json.dumps(summary)


# ---------------------------------------------------------------------------
# Value system analysis
# ---------------------------------------------------------------------------

def analyze_value_neighborhood(
    ctx: RunContext[Deps],
    token: Annotated[str, Field(description="Value token to explore, e.g. 'justice' or ' care'")],
    depth: Annotated[int, Field(description="BFS depth (1-3)", ge=1, le=3)] = 2,
) -> str:
    """Find the semantic neighborhood of a value token in the graph.

    Explores outward from nodes containing the token up to ``depth`` hops via BFS.
    Returns tokens grouped by hop distance and community context.
    """
    from collections import deque

    graph = ctx.deps.graph
    nodes = graph.find_nodes_with_token(token)
    if not nodes:
        return json.dumps({"token": token, "not_found": True})

    seed_ids = [n.id for n in nodes]
    neighborhood: dict[int, set[str]] = {d: set() for d in range(1, depth + 1)}
    visited: set[str] = set(seed_ids)
    total_visited = len(visited)

    # BFS from all seed nodes simultaneously
    queue: deque[tuple[str, int]] = deque()
    for sid in seed_ids:
        queue.append((sid, 0))

    while queue:
        current_id, dist = queue.popleft()
        if dist >= depth:
            continue
        node = graph.get_node(current_id)
        if node is None:
            continue
        for neighbor_id in node.connected_nodes:
            if neighbor_id not in visited:
                visited.add(neighbor_id)
                total_visited += 1
                next_dist = dist + 1
                neighbor_node = graph.get_node(neighbor_id)
                if neighbor_node:
                    neighborhood[next_dist].update(neighbor_node.tokens[:5])
                    queue.append((neighbor_id, next_dist))

    result = {
        "token": token,
        "seed_nodes": seed_ids,
        "neighborhood": {str(d): sorted(tokens) for d, tokens in neighborhood.items()},
        "total_nodes_visited": total_visited,
    }
    return json.dumps(result)


def compare_value_poles(
    ctx: RunContext[Deps],
    positive_tokens: Annotated[List[str], Field(description="List of positive pole tokens")],
    negative_tokens: Annotated[List[str], Field(description="List of negative pole tokens")],
) -> str:
    """Measure graph distance and community overlap between two value poles.

    Returns mean/min/max BFS distances between all (pos_node, neg_node) pairs
    (capped at 25 pairs). Also reports community overlap ratio.
    High distance + low overlap = poles are well separated.
    """
    graph = ctx.deps.graph

    pos_nodes = []
    for t in positive_tokens:
        pos_nodes.extend(graph.find_nodes_with_token(t))
    neg_nodes = []
    for t in negative_tokens:
        neg_nodes.extend(graph.find_nodes_with_token(t))

    # Deduplicate by node id
    pos_ids = list({n.id for n in pos_nodes})[:5]
    neg_ids = list({n.id for n in neg_nodes})[:5]

    if not pos_ids or not neg_ids:
        return json.dumps({
            "error": "Could not resolve tokens to graph nodes",
            "positive_found": len(pos_ids),
            "negative_found": len(neg_ids),
        })

    # BFS distances for all pairs (capped at 25)
    distances = []
    for pid in pos_ids:
        for nid in neg_ids:
            path = graph.bfs_path(pid, nid)
            if path:
                distances.append(len(path) - 1)

    # Community overlap
    community_result = graph.detect_communities()
    node_to_community: dict[str, int] = {}
    for comm_idx, comm_set in enumerate(community_result["communities"]):
        for node_id in comm_set:
            node_to_community[node_id] = comm_idx

    pos_communities = {node_to_community.get(pid) for pid in pos_ids} - {None}
    neg_communities = {node_to_community.get(nid) for nid in neg_ids} - {None}
    all_communities = pos_communities | neg_communities
    overlap = pos_communities & neg_communities
    overlap_ratio = len(overlap) / len(all_communities) if all_communities else 0.0

    result = {
        "positive_nodes": len(pos_ids),
        "negative_nodes": len(neg_ids),
        "distances": {
            "pairs_measured": len(distances),
            "mean": round(sum(distances) / len(distances), 2) if distances else None,
            "min": min(distances) if distances else None,
            "max": max(distances) if distances else None,
        },
        "community_overlap_ratio": round(overlap_ratio, 4),
    }
    return json.dumps(result)


def find_value_bridges(
    ctx: RunContext[Deps],
    value_tokens_a: Annotated[List[str], Field(description="Tokens for value cluster A")],
    value_tokens_b: Annotated[List[str], Field(description="Tokens for value cluster B")],
) -> str:
    """Identify topological bridge nodes between two value clusters.

    Finds BFS paths between clusters A and B, counts intermediate node frequency.
    Returns top 10 bridge nodes with tokens and bridge scores.
    """
    from collections import Counter

    graph = ctx.deps.graph

    nodes_a = []
    for t in value_tokens_a:
        nodes_a.extend(graph.find_nodes_with_token(t))
    nodes_b = []
    for t in value_tokens_b:
        nodes_b.extend(graph.find_nodes_with_token(t))

    a_ids = list({n.id for n in nodes_a})[:3]
    b_ids = list({n.id for n in nodes_b})[:3]

    if not a_ids or not b_ids:
        return json.dumps({
            "error": "Could not resolve tokens to graph nodes",
            "cluster_a_found": len(a_ids),
            "cluster_b_found": len(b_ids),
        })

    intermediate_counts: Counter = Counter()
    total_paths = 0

    for aid in a_ids:
        for bid in b_ids:
            path = graph.bfs_path(aid, bid)
            if path and len(path) > 2:
                total_paths += 1
                # Intermediate nodes (exclude start and end)
                for node in path[1:-1]:
                    intermediate_counts[node.id] += 1

    if not total_paths:
        return json.dumps({
            "cluster_a_nodes": a_ids,
            "cluster_b_nodes": b_ids,
            "bridges": [],
            "total_paths": 0,
        })

    top_bridges = intermediate_counts.most_common(10)
    bridges = []
    for node_id, count in top_bridges:
        node = graph.get_node(node_id)
        bridges.append({
            "node_id": node_id,
            "tokens": node.tokens[:5] if node else [],
            "count": count,
            "bridge_score": round(count / total_paths, 4),
        })

    result = {
        "cluster_a_nodes": a_ids,
        "cluster_b_nodes": b_ids,
        "total_paths": total_paths,
        "bridges": bridges,
    }
    return json.dumps(result)
