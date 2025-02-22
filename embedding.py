from transformers import GPT2Tokenizer, GPT2Model
import torch
import numpy as np
import kmapper as km
from sklearn.manifold import TSNE, Isomap
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import AgglomerativeClustering
from umap.umap_ import UMAP
import json
import gudhi
from ripser import ripser
from persim import plot_diagrams
import sys
from homology import compute_persistence, save_persistence_results
def compute_edge_weights(node_data, links, weight_type="combined"):
    """Compute edge weights based on different similarity metrics"""
    weights = {}
    
    for link in links:
        # Skip single-node links
        if len(link) < 2:
            continue
            
        source, target = str(link[0]), str(link[1])
        source_node = node_data[source]
        target_node = node_data[target]
        
        # Position-based similarity
        pos1 = np.array([source_node["position"]["x"], source_node["position"]["y"]])
        pos2 = np.array([target_node["position"]["x"], target_node["position"]["y"]])
        distance = np.linalg.norm(pos1 - pos2)
        weight = 1 / (1 + distance)  # Convert distance to similarity
        
        # Create a canonical edge key by sorting node IDs
        nodes = sorted([source, target])
        edge_key = f"{nodes[0]}|{nodes[1]}"  # Use | as separator
        weights[edge_key] = float(weight)
    
    return weights
def maxmin_sampling(points, n_samples=10000):
    """MaxMin sampling to get well-spread points"""
    n_points = len(points)
    # Start with random point
    sample_indices = [np.random.randint(n_points)]
    samples = points[sample_indices]
    
    # Iteratively add points that are furthest from existing samples
    while len(sample_indices) < n_samples:
        if len(sample_indices) % 1000 == 0:
            print(f"Selected {len(sample_indices)} points")
            
        # Compute distances to existing samples
        distances = np.min(np.linalg.norm(
            points[:, None] - samples[None, :], 
            axis=2
        ), axis=1)
        
        # Add furthest point
        next_idx = np.argmax(distances)
        sample_indices.append(next_idx)
        samples = points[sample_indices]
    
    return samples, sample_indices  # Return indices too for reference

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('gpt2')

# Get all unique tokens
tokens = tokenizer.get_vocab().keys()

# Initialize an empty dictionary to store tokens and their embeddings
token_embeddings = {}

# Get the embedding layer of the model
embedding_layer = model.get_input_embeddings()

# Convert tokens to their IDs and get their embeddings
for token in tokens:
    token_id = tokenizer.convert_tokens_to_ids(token)
    print(token)
    with torch.no_grad():
        embedding = embedding_layer(torch.tensor([token_id]))
    token_embeddings[token] = embedding.numpy()[0]
# Assume token_embeddings is a dictionary from previous steps
# Extract embeddings into a numpy array
embeddings = np.array(list(token_embeddings.values()))
isomap = PCA(n_components=100)
intermediate_embeddings = isomap.fit_transform(embeddings)
# Initialize KeplerMapper
mapper = km.KeplerMapper(verbose=1)

# Apply a dimensionality reduction technique
# Here we use PCA for simplicity
print(len(embeddings[0]))
reducer = UMAP(
    n_components=2,
    n_neighbors=100,    # Decrease from 200 to reduce connections
    min_dist=0.2,      # Increase to spread clusters more
    random_state=42    # For reproducibility
)
projected_embeddings = reducer.fit_transform(intermediate_embeddings)
plt.scatter(projected_embeddings[:,0],projected_embeddings[:,1],alpha=0.5)
plt.savefig("tsne.png")
plt.show()
print(projected_embeddings)

# Standardize the projected embeddings
scaler = StandardScaler()
projected_embeddings = scaler.fit_transform(projected_embeddings)

# Create the graph using the Mapper algorithm with modified parameters
graph = mapper.map(
    projected_embeddings, 
    embeddings,
    clusterer=AgglomerativeClustering(
        n_clusters=20,  # Decrease from 30 to create larger clusters
        linkage='complete'  # Change to 'complete' for more distinct clusters
    ),
    cover=km.Cover(
        n_cubes=30,         # Decrease from 40 to reduce overlap regions
        perc_overlap=0.3    # Decrease from 0.55 to reduce connections between cubes
    )
)

# Create a dictionary to store node information
node_data = {}

# Extract token IDs and graph structure for each node
for node in graph["nodes"]:
    # Get the indices from the node
    node_indices = graph["nodes"][node]
    
    # Get connected nodes
    connected_nodes = []
    for edge in graph["simplices"]:
        source = edge[0]  # Links are tuples or lists of [source, target]
        target = edge[ (1 if len(edge)==2 else 0)]
        if str(source) == str(node):
            connected_nodes.append(str(target))
        elif str(target) == str(node):
            connected_nodes.append(str(source))
    
    # Get node position from projected embeddings
    node_center = np.mean(projected_embeddings[node_indices], axis=0)
    
    # Store all information in the dictionary
    node_data[str(node)] = {
        "token_ids": [int(i) for i in node_indices],
        "tokens": [tokenizer.decode([tid]) for tid in node_indices],
        "size": len(node_indices),
        "connected_nodes": connected_nodes,
        "position": {
            "x": float(node_center[0]),
            "y": float(node_center[1])
        },
        "cluster_spread": {
            "std_x": float(np.std(projected_embeddings[node_indices, 0])),
            "std_y": float(np.std(projected_embeddings[node_indices, 1]))
        }
    }

# After creating the graph with mapper.map()
nodes = graph["nodes"]  # Get nodes from mapper graph

# Create the graph data structure
graph_data = {
    "nodes": nodes,
    "links": [simp for simp in graph["simplices"]],
    "metadata": {
        "total_nodes": len(graph["nodes"]),
        "total_edges": len(graph["links"]),
        "projection_bounds": {
            "x": [float(np.min(projected_embeddings[:,0])), float(np.max(projected_embeddings[:,0]))],
            "y": [float(np.min(projected_embeddings[:,1])), float(np.max(projected_embeddings[:,1]))]
        },
        "mapper_params": {
            "n_cubes": 50,
            "overlap": 0.2,
            "clusters_per_bin": 10
        }
    }
}

# Now compute edge weights after nodes and links are defined
graph_data["edge_weights"] = compute_edge_weights(node_data, graph_data["links"], weight_type="combined")

# Get the cluster centers for each node
cluster_centers = []
cluster_tokens = []
for node in graph["nodes"]:
    node_indices = graph["nodes"][node]
    center = np.mean(embeddings[node_indices], axis=0)
    cluster_centers.append(center)
    cluster_tokens.append(node_indices)

cluster_centers = np.array(cluster_centers)

# After computing embeddings...
#persistence_results = compute_persistence(embeddings)
#save_persistence_results(persistence_results)

# Save updated graph data
with open('node_clusters_with_weights.json', 'w', encoding='utf-8') as f:
    json.dump(graph_data, f, indent=2)

# Visualize the graph
tooltips = []  # Create list for all data points
for i in range(len(embeddings)):
    tooltips.append("Token: " + tokenizer.decode([i]))  # Default tooltip shows token

# Update tooltips for clustered points
for node in graph["nodes"]:
    node_indices = graph["nodes"][node]
    node_tokens = [tokenizer.decode([tid]) for tid in node_indices]
    token_string = " | ".join(node_tokens)
    
    # Update tooltip for each index in this node
    for idx in node_indices:
        tooltips[idx] = f"""
            Node: {node}<br>
            Token: {tokenizer.decode([idx])}<br>
            Cluster Size: {len(node_indices)}<br>
            Sample Tokens: {', '.join(node_tokens[:5])}
        """

# Visualization with numpy array of tooltips
mapper.visualize(
    graph,
    path_html="mapper_graph_with_weights.html",
    title="GPT2 embedding Analysis",
    custom_tooltips=np.array(tooltips),  # Convert to numpy array
    include_searchbar=True,
)
