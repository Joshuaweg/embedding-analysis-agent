import json
import numpy as np
import networkx as nx
from transformers import GPT2Tokenizer
from random import sample, randint

def create_test_graph(num_nodes=7, num_edges=10):
    """Create a test graph with randomly selected nodes and edges."""
    # Load tokenizer for sample tokens
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Create random node positions in a 2D space
    positions = []
    for _ in range(num_nodes):
        x = np.random.uniform(-1, 1)
        y = np.random.uniform(-1, 1)
        positions.append((x, y))
    
    # Create node IDs
    node_ids = []
    for i in range(num_nodes):
        cube_num = i // 2
        cluster_num = i % 2
        node_id = f"cube{cube_num}_cluster{cluster_num}"
        node_ids.append(node_id)
    
    print("\nCreated nodes:")
    for node_id in node_ids:
        print(f"- {node_id}")
    
    # Create random edges ensuring connectivity
    edges = []
    connected_nodes = {node_id: [] for node_id in node_ids}
    
    # First ensure all nodes are connected in a path
    for i in range(num_nodes - 1):
        source = node_ids[i]
        target = node_ids[i + 1]
        edges.append([source, target])
        connected_nodes[source].append(target)
        connected_nodes[target].append(source)
        print(f"Adding path edge: {source} -> {target}")
    
    # Add remaining random edges
    remaining_edges = num_edges - (num_nodes - 1)
    while remaining_edges > 0:
        # Select two random nodes
        source, target = sample(node_ids, 2)
        if target not in connected_nodes[source]:  # Only add if not already connected
            edges.append([source, target])
            connected_nodes[source].append(target)
            connected_nodes[target].append(source)
            print(f"Adding random edge: {source} -> {target}")
            remaining_edges -= 1
    
    # Create nodes dictionary
    nodes = {}
    for i, node_id in enumerate(node_ids):
        # Generate random tokens
        num_tokens = randint(5, 20)
        token_ids = np.random.randint(0, 50000, size=num_tokens).tolist()
        
        nodes[node_id] = {
            "token_ids": token_ids,
            "tokens": [tokenizer.decode([tid]) for tid in token_ids],
            "size": num_tokens,
            "connected_nodes": sorted(connected_nodes[node_id]),
            "position": {
                "x": float(positions[i][0]),
                "y": float(positions[i][1])
            },
            "cluster_spread": {
                "std_x": float(np.random.uniform(0.01, 0.05)),
                "std_y": float(np.random.uniform(0.01, 0.05))
            }
        }
    
    # Create the graph data structure
    graph_data = {
        "nodes": nodes,
        "links": edges,
        "metadata": {
            "total_nodes": num_nodes,
            "total_edges": len(edges),
            "projection_bounds": {
                "x": [-1.0, 1.0],
                "y": [-1.0, 1.0]
            },
            "mapper_params": {
                "n_cubes": 25,
                "overlap": 0.2,
                "clusters_per_bin": 2
            }
        }
    }
    
    # Verify graph consistency
    verify_graph_consistency(graph_data)
    
    # Save to file
    with open('test_graph.json', 'w') as f:
        json.dump(graph_data, f, indent=2)
    
    print("\nNode connections in final graph:")
    for node_id, node_data in nodes.items():
        print(f"{node_id} -> {node_data['connected_nodes']}")

def verify_graph_consistency(graph_data):
    """Verify that links and connected_nodes are consistent"""
    nodes = graph_data["nodes"]
    links = graph_data["links"]
    
    print("\nVerifying graph consistency:")
    
    # Convert links to a set of frozensets for easier comparison
    link_pairs = {frozenset(link) for link in links}
    
    # Check each node's connected_nodes against links
    for node_id, node_data in nodes.items():
        # Get all links involving this node
        node_links = {frozenset([node_id, other]) for other in node_data["connected_nodes"]}
        
        # Check if all connected_nodes have corresponding links
        missing_links = node_links - link_pairs
        if missing_links:
            print(f"ERROR: Node {node_id} has connections without links: {missing_links}")
        
        # Check if all links have corresponding connected_nodes
        node_links_from_links = {frozenset([node_id, other]) for link in links 
                               if node_id in link for other in link if other != node_id}
        missing_connections = node_links_from_links - node_links
        if missing_connections:
            print(f"ERROR: Node {node_id} is missing connections from links: {missing_connections}")
        
        # Check for bidirectional connections
        for connected in node_data["connected_nodes"]:
            if node_id not in nodes[connected]["connected_nodes"]:
                print(f"ERROR: One-way connection: {node_id} -> {connected}")

if __name__ == "__main__":
    create_test_graph(20, 50) 