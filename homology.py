import numpy as np
from ripser import ripser
from persim import plot_diagrams
import matplotlib.pyplot as plt
from umap.umap_ import UMAP
from transformers import GPT2Tokenizer, GPT2Model
from typing import Tuple, List, Dict
import json
import torch
import gudhi
import sys
rips_complex = None 
def grid_sampling(points, n_samples):
    """Sample points using a grid-based approach"""
    from sklearn.preprocessing import MinMaxScaler
    
    # Scale points to [0,1] cube
    scaler = MinMaxScaler()
    scaled_points = scaler.fit_transform(points)
    
    # Determine grid size
    grid_size = int(np.ceil(np.power(n_samples, 1/3)))  # for 3D
    
    # Create grid cells and find closest points
    samples = []
    for x in np.linspace(0, 1, grid_size):
        for y in np.linspace(0, 1, grid_size):
            for z in np.linspace(0, 1, grid_size):
                cell_center = np.array([x, y, z])
                # Find closest point to cell center
                distances = np.linalg.norm(scaled_points - cell_center, axis=1)
                closest_idx = np.argmin(distances)
                samples.append(closest_idx)
    
    return points[samples], samples
def repulsion_sampling(points, n_samples, iterations=3):
    """Random sampling with repulsion forces"""
    # Initial random sampling
    indices = np.random.choice(len(points), n_samples, replace=False)
    samples = points[indices]
    
    # Apply repulsion to spread points
    for _ in range(iterations):
        # Compute pairwise distances
        dists = np.linalg.norm(
            samples[:, None] - samples[None, :], 
            axis=2
        )
        # Add small repulsion force
        np.fill_diagonal(dists, np.inf)
        # Move points away from close neighbors
        forces = 1.0 / (dists + 1e-6)
        samples += np.mean(forces[:, :, None] * (samples[:, None] - samples[None, :]), axis=1)
    
    return samples, indices
def maxmin_sampling(points: np.ndarray, n_samples: int = 10000, cache_file: str = "maxmin_samples.npz") -> Tuple[np.ndarray, List[int]]:
    """MaxMin sampling with caching to avoid recomputation.
    
    Args:
        points: Input points array
        n_samples: Number of samples to select
        cache_file: File to save/load sampled indices
        
    Returns:
        Tuple of (sampled_points, sample_indices)
    """
    # Try to load cached samples first
    try:
        print("Looking for cached samples...")
        cached = np.load(cache_file)
        
        return points[cached['indices']], cached['indices'].tolist()
    except:
        print("No valid cache found, computing samples...")
    
    # Compute new samples
    n_points = len(points)
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
    
    # Save the samples
    print("Saving samples to cache...")
    np.savez(
        cache_file,
        indices=np.array(sample_indices),
        points_shape=points.shape
    )
    
    return samples, sample_indices

def compute_persistence(embeddings: np.ndarray, 
                       n_samples: int = 10000,
                       n_components: int = 3,
                       save_plot: bool = True) -> Dict:
    """Compute persistent homology using Ripser."""
    # Convert PyTorch tensor to NumPy if needed
    if torch.is_tensor(embeddings):
        embeddings = embeddings.detach().cpu().numpy()
    
    print("Computing UMAP projection...")
    persistence_reducer = UMAP(
        n_components=n_components,
        n_neighbors=500,
        min_dist=0.01,
    )
    projected_embeddings = persistence_reducer.fit_transform(embeddings)
    print(f"UMAP projection shape: {projected_embeddings.shape}")
    
    print("Performing grid sampling...")
    sampled_embeddings, sample_indices = maxmin_sampling(
        projected_embeddings, 
        n_samples=n_samples
    )
    
    # Normalize the embeddings to [-1, 1] range
    max_abs = np.max(np.abs(sampled_embeddings))
    sampled_embeddings = sampled_embeddings / max_abs
    
    print(f"Computing persistent homology on {len(sampled_embeddings)} points...")
    try:
        # Compute distance matrix first to better manage memory
        print("Computing distance matrix...")
        distances = np.linalg.norm(
            sampled_embeddings[:, None] - sampled_embeddings[None, :],
            axis=2
        )
        
        print("Running Ripser...")
        rips_complex = ripser(
            distances,
            maxdim=2,
            distance_matrix=True,
            thresh=0.07,
            coeff=2,  # Use Z/2Z coefficients
            do_cocycles=True  # Add this back to get the cocycle information
        )
        
        diagrams = rips_complex['dgms']
        cocycles = rips_complex.get('cocycles', [[], [], []])  # Default empty lists if no cocycles
        print("Persistent homology computation successful")
        
        # Add debug print
        print(f"Number of cocycles found: {[len(c) for c in cocycles]}")
        
        if save_plot:
            print("Saving persistence diagrams...")
            plt.figure(figsize=(10, 10))
            plot_diagrams(diagrams, show=False)
            plt.title("Persistence Diagram of Token Embeddings")
            plt.savefig("persistence_diagrams.png")
            plt.close()
        
        persistence_stats = {
            f"dimension_{dim}": {
                "num_features": len(diagram),
                "avg_lifetime": float(np.mean(diagram[:, 1] - diagram[:, 0])) if len(diagram) > 0 else 0,
                "max_lifetime": float(np.max(diagram[:, 1] - diagram[:, 0])) if len(diagram) > 0 else 0
            }
            for dim, diagram in enumerate(diagrams)
        }
        
        return {
            "diagrams": diagrams,
            "cocycles": cocycles,
            "sample_indices": sample_indices,
            "statistics": persistence_stats
        }
        
    except Exception as e:
        print(f"Error in Ripser computation: {str(e)}")
        import traceback
        print("Full error:")
        print(traceback.format_exc())
        raise

def save_persistence_results(results: Dict, filename: str = "persistence_results.json"):
    """Save persistence analysis results to JSON file."""
    # Convert numpy arrays and types to JSON-serializable formats
    json_results = {
        "sample_indices": [int(i) for i in results["sample_indices"]],
        "statistics": {
            dim: {
                "num_features": int(stats["num_features"]),
                "avg_lifetime": float(stats["avg_lifetime"]),
                "max_lifetime": float(stats["max_lifetime"])
            }
            for dim, stats in results["statistics"].items()
        },
        "diagrams": [diagram.tolist() for diagram in results["diagrams"]],
        # Handle cocycles differently since they're already lists
        "cocycles": results["cocycles"]
    }
    
    with open(filename, 'w') as f:
        json.dump(json_results, f, indent=2)

def compute_persistence_gudhi(sampled_embeddings):
    # Create a Rips complex
    rips = gudhi.RipsComplex(points=sampled_embeddings, max_edge_length=0.3)
    
    # Create the simplicial complex up to dimension 2
    simplex_tree = rips.create_simplex_tree(max_dimension=2)
    
    # Compute persistence
    persistence = simplex_tree.persistence()
    
    # Convert to diagram format
    diagrams = []
    for dim in range(3):  # 0,1,2 dimensions
        diagram = np.array([[p[1][0], p[1][1] if p[1][1] != float('inf') else -1] 
                           for p in persistence if p[0] == dim])
        diagrams.append(diagram)
    
    return diagrams

def extract_feature_indices(diagrams, cocycles, sample_indices, dimension, 
                          min_birth=0.01, max_birth=0.055,
                          min_death=0.032, max_death=0.055):
    """Extract indices of points that form features in the specified range."""
    diagram = diagrams[dimension]
    features_cocycles = cocycles[dimension] if dimension < len(cocycles) else []
    
    print(f"Processing dimension {dimension}")
    print(f"Found {len(features_cocycles)} cocycles")
    
    # Filter features in the specified range
    features = []
    for i, (birth, death) in enumerate(diagram):
        if death == np.inf:  # Skip infinite death times
            continue
        if (min_birth <= birth <= max_birth and 
            min_death <= death <= max_death):
            try:
                # Get the actual point indices for this feature
                feature_points = features_cocycles[i] if i < len(features_cocycles) else []
                # Map back to original token indices
                token_indices = [int(sample_indices[idx]) for idx in feature_points]
                
                features.append({
                    "birth": float(birth),
                    "death": float(death),
                    "persistence": float(death - birth),
                    "feature_index": i,
                    "token_indices": token_indices
                })
            except Exception as e:
                print(f"Error processing feature {i}: {str(e)}")
                continue
    
    return features

if __name__ == "__main__":
    # Example usage
    cached = np.load("maxmin_samples.npz")
    print("Loading embeddings...")
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2Model.from_pretrained('gpt2')
    # Convert to NumPy immediately
    embeddings = model.get_input_embeddings().weight.detach().cpu().numpy()
    print(embeddings.shape)
    results = compute_persistence(embeddings)
    save_persistence_results(results)
    print("Persistent homology analysis complete!")
    
    print("Extracting topological features...")
    
    # Extract features in specific ranges
    h1_features = extract_feature_indices(
        diagrams=results['diagrams'],
        cocycles=results['cocycles'],
        sample_indices=results['sample_indices'],
        dimension=1,  # H1 features (loops)
        min_birth=0.01,
        max_birth=0.055,
        min_death=0.032,
        max_death=0.055
    )
    
    h2_features = extract_feature_indices(
        diagrams=results['diagrams'],
        cocycles=results['cocycles'],
        sample_indices=results['sample_indices'],
        dimension=2,  # H2 features (voids)
        min_birth=0.01,
        max_birth=0.055,
        min_death=0.032,
        max_death=0.055
    )
    
    # Save the feature information
    feature_results = {
        "h1_features": h1_features,
        "h2_features": h2_features
    }
    
    with open('persistence_features.json', 'w') as f:
        json.dump(feature_results, f, indent=2)
    
    print(f"\nFound {len(h1_features)} H1 features and {len(h2_features)} H2 features")
    print("Features saved to persistence_features.json")
    
    # Print example features with their tokens
    print("\nExample H1 features (first 3):")
    for i, feature in enumerate(h1_features[:3]):
        print(f"\nFeature {i+1}:")
        print(f"Birth: {feature['birth']:.3f}")
        print(f"Death: {feature['death']:.3f}")
        print(f"Persistence: {feature['persistence']:.3f}")
        print("Tokens involved:")
        for idx in feature['token_indices']:
            print(f"  {idx}: {tokenizer.decode([idx])}")

    print("\nExample H2 features (first 3):")
    for i, feature in enumerate(h2_features[:3]):
        print(f"\nFeature {i+1}:")
        print(f"Birth: {feature['birth']:.3f}")
        print(f"Death: {feature['death']:.3f}")
        print(f"Persistence: {feature['persistence']:.3f}")
        print(f"Feature Index: {feature['feature_index']}")
        print(f"Token Indices: {', '.join(map(str, feature['token_indices']))}") 