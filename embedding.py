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
    n_neighbors=200,    # Increase to consider more global structure
    min_dist=0.1,     # Decrease to allow tighter clusters
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
        n_clusters=30,
        linkage='average'
    ),
    cover=km.Cover(
        n_cubes=40,
        perc_overlap=0.55
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

# Add global graph metadata
graph_data = {
    "nodes": node_data,
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
with open('node_clusters_2.json', 'w', encoding='utf-8') as f:
    json.dump(graph_data, f, indent=2)




# Visualize the graph
node_data = {}
for node in graph["nodes"]:
    node_indices = graph["nodes"][node]
    
    # Convert indices to actual tokens and join them
    node_tokens = [tokenizer.decode([tid]) for tid in node_indices]
    token_string = " | ".join(node_tokens)  # Join tokens with separator
    
    # Get connected nodes
    connected_nodes = []
    for edge in graph["simplices"]:
        source = edge[0]
        target = edge[(1 if len(edge)==2 else 0)]
        if str(source) == str(node):
            connected_nodes.append(str(target))
        elif str(target) == str(node):
            connected_nodes.append(str(source))
    
    node_center = np.mean(projected_embeddings[node_indices], axis=0)
    
    # Update node_data structure
    node_data[str(node)] = {
        "members": token_string,  # This is the key change - use token_string instead of token_ids
        "tokens": node_tokens,    # Keep the individual tokens as well
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

# Then in your visualization call, you might want to customize the tooltip
mapper.visualize(
    graph,
    path_html="mapper_graph_2.html",
    title="GPT2 embedding Analysis",
    custom_tooltips=node_data  # This should display the tokens in the tooltips
)
