import numpy as np
import plotly.graph_objects as go
from transformers import GPT2Tokenizer, GPT2Model
from umap import UMAP
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import json
import torch

def load_samples_and_embeddings():
    """Load cached samples and their embeddings."""
    # Load the cached samples
    cached = np.load("maxmin_samples.npz")
    sample_indices = cached['indices']
    
    # Load the original embeddings
    print("Loading GPT-2 embeddings...")
    model = GPT2Model.from_pretrained('gpt2')
    embeddings = model.get_input_embeddings().weight.detach().cpu().numpy()
    
    # Get embeddings for sampled indices
    sampled_embeddings = embeddings[sample_indices]
    
    # Perform UMAP reduction to 3D
    print("Performing UMAP reduction to 3D...")
    print("Computing UMAP projection...")
    reducer = UMAP(
        n_components=3,
        n_neighbors=500,
        min_dist=0.01,
    )
    coords_3d = reducer.fit_transform(sampled_embeddings)
    
    # Load features
    with open('persistence_features.json', 'r') as f:
        features = json.load(f)
    
    with open('persistence_results.json', 'r') as f:
        persistence = json.load(f)
    
    return sample_indices, coords_3d, features, persistence

def create_3d_visualization(sample_indices, coords, features, persistence, 
                          highlight_h1=True, highlight_h2=True):
    """Create interactive 3D visualization of sampled points."""
    # Load tokenizer for labels
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Create the base scatter plot
    fig = go.Figure()
    
    # Add all points in gray
    fig.add_trace(go.Scatter3d(
        x=coords[:, 0],
        y=coords[:, 1],
        z=coords[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color='gray',
            opacity=0.5
        ),
        name='All Points',
        text=[tokenizer.decode([idx]) for idx in sample_indices],
        hoverinfo='text'
    ))
    
    # Highlight H1 features (loops) in orange
    if highlight_h1 and 'h1_features' in features:
        h1_indices = [idx for feature in features['h1_features'] for idx in feature]
        if h1_indices:  # Only if we have H1 features
            h1_points = coords[h1_indices]
            fig.add_trace(go.Scatter3d(
                x=h1_points[:, 0],
                y=h1_points[:, 1],
                z=h1_points[:, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color='orange',
                    opacity=0.8
                ),
                name='H1 Features (Loops)',
                text=[tokenizer.decode([sample_indices[idx]]) for idx in h1_indices],
                hoverinfo='text'
            ))
    
    # Highlight H2 features (voids) in green
    if highlight_h2 and 'h2_features' in features:
        h2_indices = [idx for feature in features['h2_features'] for idx in feature]
        if h2_indices:  # Only if we have H2 features
            h2_points = coords[h2_indices]
            fig.add_trace(go.Scatter3d(
                x=h2_points[:, 0],
                y=h2_points[:, 1],
                z=h2_points[:, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color='green',
                    opacity=0.8
                ),
                name='H2 Features (Voids)',
                text=[tokenizer.decode([sample_indices[idx]]) for idx in h2_indices],
                hoverinfo='text'
            ))
    
    # Update layout
    fig.update_layout(
        title='3D UMAP Projection of Token Embeddings with Topological Features',
        scene=dict(
            xaxis_title='UMAP 1',
            yaxis_title='UMAP 2',
            zaxis_title='UMAP 3'
        ),
        width=1200,
        height=1000,
        showlegend=True
    )
    
    # Save to HTML for interactive viewing
    fig.write_html("embedding_visualization_3d_enhanced.html")
    
    return fig

if __name__ == "__main__":
    # Load the data
    sample_indices, coords_3d, features, persistence = load_samples_and_embeddings()
    
    # Create visualization
    fig = create_3d_visualization(sample_indices, coords_3d, features, persistence)
    print("Visualization saved to embedding_visualization_3d.html") 