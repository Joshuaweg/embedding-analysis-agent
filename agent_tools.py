from typing import List, Dict, Any, Optional
from graph_structure import TokenGraph
from agno.tools import tool
import json

class TokenGraphTools:
    def __init__(self, graph: TokenGraph):
        self.graph = graph

    @tool(show_result=True)
    def get_node_tool(self, node_id: str) -> str:
        """Get information about a specific node in the graph.
        
        Args:
            node_id: The ID of the node to get information about
            
        Returns:
            Formatted string containing node information
        """
        return self.graph.get_node_info(node_id)

    @tool(show_result=True)
    def get_connected_nodes_tool(self, node_id: str) -> List[str]:
        """Get list of nodes connected to a specific node.
        
        Args:
            node_id: The ID of the node to find connections for
            
        Returns:
            List of connected node IDs
        """
        connected_nodes = self.graph.get_connected_nodes(node_id)
        return [node.id for node in connected_nodes]

    @tool(show_result=True)
    def get_hypercube_nodes_tool(self, cube_num: int) -> List[str]:
        """Get all nodes in a specific hypercube.
        
        Args:
            cube_num: The hypercube number to get nodes from
            
        Returns:
            List of node IDs in the specified hypercube
        """
        nodes = self.graph.get_hypercube_nodes(cube_num)
        return [node.id for node in nodes]

    @tool(show_result=True)
    def find_nodes_with_token_tool(self, token: str) -> List[str]:
        """Find all nodes that contain a specific token.
        
        Args:
            token: The token to search for
            
        Returns:
            List of node IDs containing the token
        """
        nodes = self.graph.find_nodes_with_token(token)
        return [node.id for node in nodes]

    @tool(show_result=True)
    def bfs_path_tool(self, start_node_id: str, end_node_id: str) -> List[str]:
        """Find shortest path between two nodes using breadth-first search.
        
        Args:
            start_node_id: ID of the starting node
            end_node_id: ID of the target node
            
        Returns:
            List of node IDs representing the path
        """
        path = self.graph.bfs_path(start_node_id, end_node_id)
        return [node.id for node in path]

    @tool(show_result=True)
    def find_all_paths_tool(self, start_node_id: str, end_node_id: str, max_depth: int = 5) -> List[List[str]]:
        """Find all paths between two nodes up to a maximum depth.
        
        Args:
            start_node_id: ID of the starting node
            end_node_id: ID of the target node
            max_depth: Maximum path length to consider
            
        Returns:
            List of paths, where each path is a list of node IDs
        """
        paths = self.graph.find_all_paths(start_node_id, end_node_id, max_depth)
        return [[node.id for node in path] for path in paths]

    @tool(show_result=True)
    def analyze_components_tool(self) -> Dict[str, Any]:
        """Analyze connected components in the graph.
        
        Returns:
            Dictionary containing component analysis results
        """
        return self.graph.analyze_components()

    @tool(show_result=True)
    def random_walk_tool(self, start_node_id: Optional[str] = None, num_steps: int = 5) -> List[str]:
        """Perform a random walk through the graph.
        
        Args:
            start_node_id: Starting node ID (if None, randomly selected)
            num_steps: Number of steps to take in the walk
            
        Returns:
            List of node IDs representing the walk path
        """
        walk = self.graph.random_walk(start_node_id, num_steps)
        return [node.id for node in walk]

    @tool(show_result=True)
    def analyze_network_tool(self) -> Dict[str, Any]:
        """Perform social network analysis on the graph structure.
        
        Returns:
            Dictionary containing network analysis results
        """
        return self.graph.analyze_network()

    @tool(show_result=True)
    def detect_communities_tool(self) -> Dict[str, Any]:
        """Detect communities in the graph using Louvain method.
        
        Returns:
            Dictionary containing community detection results
        """
        return self.graph.detect_communities()

    @tool(show_result=True)
    def compute_node_centrality_tool(self) -> Dict[str, Any]:
        """Compute various centrality metrics for nodes.
        
        Returns:
            Dictionary containing centrality metrics
        """
        return self.graph.compute_node_centrality()

    @tool(show_result=True)
    def extract_subgraph_tool(self, node_ids: List[str]) -> Dict[str, Any]:
        """Extract subgraph containing specified nodes and their neighbors.
        
        Args:
            node_ids: List of node IDs to include in subgraph
            
        Returns:
            Dictionary containing subgraph nodes
        """
        return self.graph.extract_subgraph(node_ids)

    @tool(show_result=True)
    def analyze_token_patterns_tool(self) -> Dict[str, Any]:
        """Analyze patterns in token distributions across nodes.
        
        Returns:
            Dictionary containing token pattern analysis results
        """
        return self.graph.analyze_token_patterns()

    @tool(show_result=True)
    def analyze_paths_tool(self, start_node_id: str, max_length: int = 5) -> List[List[str]]:
        """Analyze all paths starting from a node up to max_length.
        
        Args:
            start_node_id: ID of the starting node
            max_length: Maximum path length to consider
            
        Returns:
            List of paths, where each path is a list of node IDs
        """
        return self.graph.analyze_paths(start_node_id, max_length)

    @tool(show_result=True)
    def compute_graph_statistics_tool(self) -> Dict[str, Any]:
        """Compute various statistical measures of the graph.
        
        Returns:
            Dictionary containing graph statistics
        """
        return self.graph.compute_graph_statistics()

    @tool(show_result=True)
    def weighted_bfs_path_tool(self, start_node_id: str, end_node_id: str) -> List[str]:
        """Find path optimizing for edge weights.
        
        Args:
            start_node_id: ID of the starting node
            end_node_id: ID of the target node
            
        Returns:
            List of node IDs representing the path
        """
        return self.graph.weighted_bfs_path(start_node_id, end_node_id)
