from dataclasses import dataclass
from typing import List, Dict, Set
import json
from collections import deque
from transformers import GPT2Tokenizer
from random import choice, random
import networkx as nx
from collections import Counter
import numpy as np

@dataclass
class Node:
    """Represents a cluster node in the graph"""
    id: str                     # e.g., "cube0_cluster0"
    token_ids: List[int]       # List of token IDs in this cluster
    tokens: List[str]          # Actual token strings
    size: int                  # Number of tokens in cluster
    position: Dict[str, float] # x,y coordinates
    spread: Dict[str, float]   # std_x, std_y of cluster spread
    connected_nodes: Set[str]  # Set of connected node IDs

class TokenGraph:
    """Represents the entire token cluster graph using adjacency lists"""
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        
    @classmethod
    def from_json(cls, json_file: str) -> 'TokenGraph':
        """Create a TokenGraph from the node_clusters.json file"""
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        graph = cls()
        
        # Create nodes from the JSON data
        for node_id, node_info in data["nodes"].items():
            node = Node(
                id=node_id,
                token_ids=node_info["token_ids"],
                tokens=node_info["tokens"],
                size=node_info["size"],
                position=node_info["position"],
                spread=node_info["cluster_spread"],
                connected_nodes=set(node_info["connected_nodes"])
            )
            graph.nodes[node_id] = node
            
        return graph
    
    def get_node(self, node_id: str) -> Node:
        """Get a node by its ID"""
        return self.nodes.get(node_id)
    
    def get_connected_nodes(self, node_id: str) -> List[Node]:
        """Get list of nodes connected to the given node"""
        node = self.get_node(node_id)
        if not node:
            return []
        return [self.nodes[nid] for nid in node.connected_nodes]
    
    def get_hypercube_nodes(self, cube_num: int) -> List[Node]:
        """Get all nodes in a specific hypercube"""
        prefix = f"cube{cube_num}_"
        return [node for node in self.nodes.values() if node.id.startswith(prefix)]
    
    def find_nodes_with_token(self, token: str) -> List[Node]:
        """Find all nodes that contain a specific token"""
        return [node for node in self.nodes.values() if token in node.tokens]
    
    def get_node_info(self, node_id: str) -> str:
        """Get a formatted string of information about a node"""
        node = self.get_node(node_id)
        if not node:
            return f"Node {node_id} not found"
        
        info = [
            f"Node: {node.id}",
            f"Size: {node.size} tokens",
            f"Position: (x={node.position['x']:.3f}, y={node.position['y']:.3f})",
            f"Connected to: {', '.join(sorted(node.connected_nodes))}",
            "\nSample tokens:",
            "-" * 30
        ]
        
        # Add sample of tokens (first 10)
        for i, token in enumerate(node.tokens[:10]):
            info.append(f"{node.token_ids[i]}: {token}")
        if len(node.tokens) > 10:
            info.append("...")
            
        return "\n".join(info)

    def bfs_path(self, start_node_id: str, end_node_id: str) -> List[Node]:
        """Find shortest path between two nodes using breadth-first search.
        
        Args:
            start_node_id: ID of the starting node
            end_node_id: ID of the target node
            
        Returns:
            List of Node objects representing the path, or empty list if no path exists
        """
        if start_node_id not in self.nodes or end_node_id not in self.nodes:
            return []
        
        # Queue will store tuples of (node_id, path_to_node)
        queue = deque([(start_node_id, [start_node_id])])
        visited = {start_node_id}
        
        while queue:
            current_id, path = queue.popleft()
            
            # Check if we've reached the target
            if current_id == end_node_id:
                return [self.nodes[node_id] for node_id in path]
            
            # Add all unvisited neighbors to the queue
            current_node = self.nodes[current_id]
            for neighbor_id in current_node.connected_nodes:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    new_path = path + [neighbor_id]
                    queue.append((neighbor_id, new_path))
        
        return []  # No path found

    def find_all_paths(self, start_node_id: str, end_node_id: str, max_depth: int = 5) -> List[List[Node]]:
        """Find all paths between two nodes up to a maximum depth.
        
        Args:
            start_node_id: ID of the starting node
            end_node_id: ID of the target node
            max_depth: Maximum path length to consider
            
        Returns:
            List of paths, where each path is a list of Node objects
        """
        def dfs(current_id: str, target_id: str, path: List[str], visited: Set[str], paths: List[List[str]]):
            if len(path) > max_depth:
                return
            
            if current_id == target_id:
                paths.append(path[:])
                return
            
            for neighbor_id in self.nodes[current_id].connected_nodes:
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    dfs(neighbor_id, target_id, path + [neighbor_id], visited, paths)
                    visited.remove(neighbor_id)
        
        if start_node_id not in self.nodes or end_node_id not in self.nodes:
            return []
        
        path_lists = []
        dfs(start_node_id, end_node_id, [start_node_id], {start_node_id}, path_lists)
        
        # Convert path lists of IDs to lists of Node objects
        return [[self.nodes[node_id] for node_id in path] for path in path_lists]

    def get_path_info(self, path: List[Node]) -> str:
        """Get a formatted string describing a path through the graph.
        
        Args:
            path: List of Node objects representing a path
            
        Returns:
            Formatted string describing the path
        """
        if not path:
            return "No path found"
        
        info = ["Path through graph:"]
        info.append("-" * 40)
        
        for i, node in enumerate(path):
            info.append(f"\nStep {i+1}: {node.id}")
            info.append(f"Size: {node.size} tokens")
            info.append("Sample tokens:")
            # Show up to 5 sample tokens
            for token_id, token in zip(node.token_ids[:5], node.tokens[:5]):
                info.append(f"  {token_id}: {token}")
            if len(node.tokens) > 5:
                info.append("  ...")
            
            if i < len(path) - 1:
                info.append("\n↓")  # Show arrow to next node
        
        return "\n".join(info)

    def analyze_components(self) -> Dict:
        """Analyze connected components in the graph using BFS.
        
        Returns:
            Dict containing:
            - total_components: number of distinct components
            - component_sizes: list of (component_id, size, edge_count) tuples
            - node_components: dict mapping node_ids to their component_id
            - largest_component: (component_id, size, edge_count) of biggest component
            - components: dict mapping component_id to set of node_ids
        """
        unvisited = set(self.nodes.keys())
        node_components = {}  # Maps node_id -> component_id
        component_sizes = []  # List of (component_id, node_count, edge_count) tuples
        components = {}  # Maps component_id -> set of node_ids
        
        component_id = 0
        
        while unvisited:
            # Start a new component
            start_node = next(iter(unvisited))
            component = set()
            
            # Use BFS to find all nodes in this component
            queue = deque([start_node])
            while queue:
                node_id = queue.popleft()
                if node_id in unvisited:
                    # Add to current component
                    component.add(node_id)
                    unvisited.remove(node_id)
                    node_components[node_id] = component_id
                    
                    # Add unvisited neighbors to queue
                    node = self.nodes[node_id]
                    for neighbor_id in node.connected_nodes:
                        if neighbor_id in unvisited:
                            queue.append(neighbor_id)
            
            # Count edges in this component
            edge_count = 0
            for node_id in component:
                node = self.nodes[node_id]
                # Count edges only within this component
                edge_count += sum(1 for neighbor in node.connected_nodes 
                                if neighbor in component)
            
            # Divide by 2 since we counted each edge twice
            edge_count //= 2
            
            # Record component info
            components[component_id] = component
            component_sizes.append((component_id, len(component), edge_count))
            component_id += 1
        
        # Sort components by size (largest first)
        component_sizes.sort(key=lambda x: x[1], reverse=True)
        
        return {
            "total_components": len(component_sizes),
            "component_sizes": component_sizes,
            "node_components": node_components,
            "largest_component": component_sizes[0] if component_sizes else None,
            "components": components
        }

    def get_component_info(self, analysis: Dict) -> str:
        """Get formatted string describing component analysis results.
        
        Args:
            analysis: Dict returned by analyze_components()
            
        Returns:
            Formatted string with component information
        """
        info = ["Graph Component Analysis"]
        info.append("=" * 40)
        
        info.append(f"\nTotal Components: {analysis['total_components']}")
        
        if analysis['largest_component']:
            comp_id, size, edges = analysis['largest_component']
            info.append(f"\nLargest Component:")
            info.append(f"- ID: {comp_id}")
            info.append(f"- Nodes: {size}")
            info.append(f"- Edges: {edges}")
            info.append(f"- Edge Density: {2*edges/(size*(size-1)):.3f}")
        
        info.append("\nComponent Details:")
        for comp_id, size, edges in analysis['component_sizes']:
            density = 2*edges/(size*(size-1)) if size > 1 else 0
            info.append(f"\nComponent {comp_id}:")
            info.append(f"- Nodes: {size}")
            info.append(f"- Edges: {edges}")
            info.append(f"- Edge Density: {density:.3f}")
        
        # Add statistics
        sizes = [size for _, size, _ in analysis['component_sizes']]
        edges = [edge_count for _, _, edge_count in analysis['component_sizes']]
        
        avg_size = sum(sizes) / len(sizes) if sizes else 0
        avg_edges = sum(edges) / len(edges) if edges else 0
        
        info.append(f"\nSummary Statistics:")
        info.append(f"- Average Component Size: {avg_size:.1f} nodes")
        info.append(f"- Average Edge Count: {avg_edges:.1f} edges")
        info.append(f"- Total Components: {analysis['total_components']}")
        comp_id, size, edges = analysis['largest_component']
        info.append(f"\nLargest Component:")
        info.append(f"- ID: {comp_id}")
        info.append(f"- Nodes: {size}")
        info.append(f"- Edges: {edges}")
        info.append(f"- Edge Density: {2*edges/(size*(size-1)):.3f}")
        
        return "\n".join(info)

    def random_walk(self, start_node_id: str = None, num_steps: int = 5) -> List[Node]:
        """Perform a random walk through the graph.
        
        Args:
            start_node_id: Starting node ID (if None, randomly selected)
            num_steps: Number of steps to take in the walk
            
        Returns:
            List of Node objects representing the walk path
        """
        # If no start node provided, randomly select one
        if start_node_id is None:
            start_node_id = choice(list(self.nodes.keys()))
        elif start_node_id not in self.nodes:
            raise ValueError(f"Start node {start_node_id} not found in graph")
        
        current_node = self.nodes[start_node_id]
        walk_path = [current_node]
        
        for _ in range(num_steps):
            # Get connected nodes
            neighbors = list(current_node.connected_nodes)
            if not neighbors:
                break  # Stop if we reach a dead end
            
            # Select random neighbor
            next_node_id = choice(neighbors)
            current_node = self.nodes[next_node_id]
            walk_path.append(current_node)
        
        return walk_path

    def get_walk_info(self, walk: List[Node]) -> str:
        """Get a formatted string describing a random walk through the graph.
        
        Args:
            walk: List of Node objects from random_walk()
            
        Returns:
            Formatted string describing the walk
        """
        if not walk:
            return "No walk path found"
        
        info = ["Random Walk through Graph:"]
        info.append("-" * 40)
        
        for i, node in enumerate(walk):
            info.append(f"\nStep {i}: {node.id}")
            info.append(f"Size: {node.size} tokens")
            info.append("Sample tokens:")
            # Show up to 3 sample tokens
            for token_id, token in zip(node.token_ids[:3], node.tokens[:3]):
                info.append(f"  {token_id}: {token}")
            if len(node.tokens) > 3:
                info.append("  ...")
        
            if i < len(walk) - 1:
                info.append("\n→")  # Show arrow to next node
        
        return "\n".join(info)

    def analyze_network(self) -> Dict:
        """Perform social network analysis on the graph structure."""
        # Convert our graph structure to NetworkX format if not already
        G = nx.Graph()
        for node_id, node in self.nodes.items():
            G.add_node(node_id)
            for neighbor in node.connected_nodes:
                G.add_edge(node_id, neighbor)
        
        analysis = {
            "global_metrics": self._compute_global_metrics(G),
            "node_metrics": self._compute_node_metrics(G),
            "community_metrics": self._compute_community_metrics(G),
            "path_metrics": self._compute_path_metrics(G)
        }
        
        return analysis
    
    def _compute_global_metrics(self, G: nx.Graph) -> Dict:
        """Compute global network metrics."""
        return {
            "num_nodes": G.number_of_nodes(),
            "num_edges": G.number_of_edges(),
            "density": nx.density(G),
            "average_clustering": nx.average_clustering(G),
            "is_connected": nx.is_connected(G),
            "number_connected_components": nx.number_connected_components(G),
            "average_shortest_path_length": self._safe_average_path_length(G),
            "diameter": self._safe_diameter(G),
            "degree_assortativity": nx.degree_assortativity_coefficient(G)
        }
    
    def _compute_node_metrics(self, G: nx.Graph) -> Dict:
        """Compute node-level metrics."""
        metrics = {
            "degree_centrality": nx.degree_centrality(G),
            "betweenness_centrality": nx.betweenness_centrality(G),
            "closeness_centrality": nx.closeness_centrality(G),
            "eigenvector_centrality": self._safe_eigenvector_centrality(G),
            "clustering_coefficient": nx.clustering(G)
        }
        
        # Find hubs and authorities
        degree_dist = Counter(dict(G.degree()).values())
        hub_threshold = np.percentile(list(degree_dist.keys()), 90)
        metrics["hubs"] = [node for node, degree in G.degree() 
                          if degree >= hub_threshold]
        
        return metrics
    
    def _compute_community_metrics(self, G: nx.Graph) -> Dict:
        """Analyze community structure."""
        try:
            communities = nx.community.greedy_modularity_communities(G)
            return {
                "num_communities": len(communities),
                "modularity": nx.community.modularity(G, communities),
                "communities": [list(c) for c in communities],
                "sizes": [len(c) for c in communities]
            }
        except:
            return {"error": "Community detection failed"}
    
    def _compute_path_metrics(self, G: nx.Graph) -> Dict:
        """Analyze path-related metrics."""
        return {
            "average_shortest_path": self._safe_average_path_length(G),
            "eccentricity": self._safe_eccentricity(G),
            "radius": self._safe_radius(G),
            "diameter": self._safe_diameter(G)
        }
    
    def get_network_summary(self, analysis: Dict) -> str:
        """Generate a human-readable summary of network analysis."""
        global_metrics = analysis["global_metrics"]
        node_metrics = analysis["node_metrics"]
        community_metrics = analysis["community_metrics"]
        
        summary = ["Network Analysis Summary", "=" * 30, ""]
        
        # Global metrics
        summary.extend([
            "Global Metrics:",
            f"- Nodes: {global_metrics['num_nodes']}",
            f"- Edges: {global_metrics['num_edges']}",
            f"- Density: {global_metrics['density']:.3f}",
            f"- Average Clustering: {global_metrics['average_clustering']:.3f}",
            f"- Connected Components: {global_metrics['number_connected_components']}",
            ""
        ])
        
        # Top nodes by different centrality measures
        summary.append("Top Nodes by Centrality:")
        for metric in ["degree_centrality", "betweenness_centrality", "closeness_centrality"]:
            top_nodes = sorted(node_metrics[metric].items(), 
                             key=lambda x: x[1], reverse=True)[:5]
            summary.append(f"\n{metric.replace('_', ' ').title()}:")
            for node, score in top_nodes:
                node_obj = self.nodes[node]
                sample_tokens = ', '.join(node_obj.tokens[:3])
                summary.append(f"- {node} ({sample_tokens}...): {score:.3f}")
        
        # Community structure
        if "num_communities" in community_metrics:
            summary.extend([
                "",
                "Community Structure:",
                f"- Number of Communities: {community_metrics['num_communities']}",
                f"- Modularity: {community_metrics['modularity']:.3f}",
                "- Community Sizes: " + 
                ', '.join(str(s) for s in sorted(community_metrics['sizes'], reverse=True))
            ])
        
        return '\n'.join(summary)
    
    # Helper methods for safe computation
    def _safe_average_path_length(self, G):
        try:
            return nx.average_shortest_path_length(G)
        except:
            return float('inf')
    
    def _safe_diameter(self, G):
        try:
            return nx.diameter(G)
        except:
            return float('inf')
    
    def _safe_radius(self, G):
        try:
            return nx.radius(G)
        except:
            return float('inf')
    
    def _safe_eccentricity(self, G):
        try:
            return nx.eccentricity(G)
        except:
            return {}
    
    def _safe_eigenvector_centrality(self, G):
        try:
            return nx.eigenvector_centrality(G)
        except:
            return {}

# Example usage:
if __name__ == "__main__":
    # Load the graph
    graph = TokenGraph.from_json("node_clusters_2.json")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    word = "African American"
    # Example queries
    print("\nExample node info:")
    print(graph.get_node_info("cube79_cluster3"))
    
    print("\nNodes in hypercube 3:")
    for node in graph.get_hypercube_nodes(3):
        print(f"- {node.id} (size: {node.size})")
    
    print(f"\nNodes containing {word}:")
    tokens = tokenizer.encode(word)
    texts = [tokenizer.decode(token) for token in tokens]
    print(texts)
    for text in texts:
        print(text)
        for node in graph.find_nodes_with_token(text):
            print(f"- {node.id}")
            print(graph.get_node_info(node.id))
    
    # Test BFS path finding
    print("\nTesting BFS path finding:")
    start_node = "cube272_cluster9"
    end_node = "cube52_cluster3"
    
    print(f"\nFinding path from {start_node} to {end_node}:")
    path = graph.bfs_path(start_node, end_node)
    print(graph.get_path_info(path))
    
    # Test finding all paths
    print("\nFinding all paths (max depth 3):")
    all_paths = graph.find_all_paths(start_node, end_node, max_depth=3)
    print(f"Found {len(all_paths)} possible paths")
    
    for i, path in enumerate(all_paths, 1):
        print(f"\nPath {i}:")
        print(graph.get_path_info(path))
    
    # Test component analysis
    print("\nAnalyzing graph components:")
    component_analysis = graph.analyze_components()
    print(graph.get_component_info(component_analysis))
    
    # Example: find which component a specific node is in
    test_node = "cube272_cluster9"
    component_id = component_analysis["node_components"].get(test_node)
    if component_id is not None:
        print(f"\nNode {test_node} is in component {component_id}")
        component_size = next(size for cid, size, _ in component_analysis["component_sizes"] 
                            if cid == component_id)
        print(f"This component has {component_size} nodes")
    
    # Test random walk
    print("\nTesting Random Walk:")
    
    # Test with specific start node
    start_node = "cube52_cluster3"
    print(f"\nRandom walk from {start_node}:")
    walk = graph.random_walk(start_node, num_steps=5)
    print(graph.get_walk_info(walk))
    
    # Test with random start node
    print("\nRandom walk from random start:")
    walk = graph.random_walk(num_steps=7)
    print(graph.get_walk_info(walk))
    
    # Perform network analysis
    analysis = graph.analyze_network()
    
    # Print summary
    print(graph.get_network_summary(analysis))
    
    # Example: Find important nodes that connect different communities
    node_metrics = analysis["node_metrics"]
    high_betweenness_nodes = sorted(
        node_metrics["betweenness_centrality"].items(),
        key=lambda x: x[1],
        reverse=True
    )[:10]
    
    print("\nTop Bridge Nodes:")
    for node_id, score in high_betweenness_nodes:
        node = graph.get_node(node_id)
        print(f"\n{node_id} (Betweenness: {score:.3f})")
        print(f"Sample tokens: {', '.join(node.tokens[:5])}") 