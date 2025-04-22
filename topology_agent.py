from agno.agent import Agent, RunResponse
from agno.models.ollama import Ollama
from graph_structure import TokenGraph
from agent_tools import TokenGraphTools
import json

class TopologyAgent:
    def __init__(self, graph_path: str):
        # Initialize the graph and tools
        self.graph = TokenGraph.from_json(graph_path)
        self.tools = TokenGraphTools(self.graph)
        
        # Initialize the Agno agent with llama3.8b
        self.agent = Agent(
            model=Ollama(id="llama3.1:8b"),
            tools=[
                self.tools.get_node_tool,
                self.tools.get_connected_nodes_tool,
                self.tools.get_hypercube_nodes_tool,
                self.tools.find_nodes_with_token_tool,
                self.tools.bfs_path_tool,
                self.tools.find_all_paths_tool,
                self.tools.analyze_components_tool,
                self.tools.random_walk_tool,
                self.tools.analyze_network_tool,
                self.tools.detect_communities_tool,
                self.tools.compute_node_centrality_tool,
                self.tools.extract_subgraph_tool,
                self.tools.analyze_token_patterns_tool,
                self.tools.analyze_paths_tool,
                self.tools.compute_graph_statistics_tool,
                self.tools.weighted_bfs_path_tool
            ],
            tool_choice="auto",
            markdown=True,
            instructions="""You are a topology analysis agent specialized in analyzing and navigating token graphs.
Your capabilities include:
- Finding paths between nodes
- Analyzing graph structure and components
- Detecting communities and patterns
- Computing various graph metrics
- Performing random walks and path analysis

Use the available tools to help users analyze and understand the token graph structure.
Always provide clear explanations of your findings and reasoning.

When analyzing the graph:
1. First understand what the user wants to know
2. Select appropriate tools to gather the necessary information
3. Analyze the results and provide insights
4. Explain your reasoning and any patterns you discover
5. If relevant, suggest additional analyses that might be interesting"""
        )
    
    def analyze(self, prompt: str) -> None:
        """Analyze the graph based on the user's prompt.
        
        Args:
            prompt: The analysis prompt to send to the agent
        """
        self.agent.print_response(prompt)

# Example usage:
if __name__ == "__main__":
    # Initialize the agent
    agent = TopologyAgent("node_clusters_with_weights.json")
    # Example analyses
    agent.analyze("Hello, how are you?")
    agent.analyze("What information can you tell me about node cube79_cluster3?")
    agent.analyze("Find all nodes connected to cube79_cluster3")
    agent.analyze("Analyze the community structure of the graph")
    agent.analyze("Find the shortest path between cube79_cluster3 and cube52_cluster3")
