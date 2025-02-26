import numpy as np
import plotly.graph_objects as go

def whiten_data(x, y, z):
    """Apply whitening transformation to make data more spherical"""
    # Combine data
    data = np.vstack([x, y, z]).T
    
    # Calculate mean and center data
    mean = np.mean(data, axis=0)
    centered_data = data - mean
    
    # Calculate covariance matrix and its eigendecomposition
    cov = np.cov(centered_data.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Prevent division by zero
    eigenvalues = np.maximum(eigenvalues, 1e-6)
    
    # Whitening transformation
    whitening_matrix = eigenvectors @ np.diag(1.0/np.sqrt(eigenvalues)) @ eigenvectors.T
    whitened_data = centered_data @ whitening_matrix
    
    # Scale back to original range
    scale = 5.0 / np.max(np.abs(whitened_data))
    whitened_data *= scale
    
    return whitened_data[:, 0], whitened_data[:, 1], whitened_data[:, 2]

def generate_3d_points(n_points=5000):
    """Generate random points in 3D space"""
    # Generate random x,y coordinates with Gaussian distribution
    x = np.random.normal(0, 2, n_points)
    y = np.random.normal(0, 2, n_points)
    z = np.random.normal(0, 2, n_points)
    
    # Apply whitening
    x, y, z = whiten_data(x, y, z)
    
    return x, y, z

def classify_points(x, y, z, overlap=0.08):
    """Classify points based on z = x + y function with overlap band"""
    # Calculate expected z value for each point
    z_function = x + y
    
    # Create overlap band
    band_width = (np.max(z) - np.min(z)) * overlap
    upper_bound = z_function + band_width/2
    lower_bound = z_function - band_width/2
    
    # Initial classification
    above_function = z > upper_bound
    below_function = z < lower_bound
    overlap_mask = ~(above_function | below_function)
    
    # Randomly assign overlap points to above or below
    overlap_size = np.sum(overlap_mask)
    random_assignment = np.random.random(overlap_size) > 0.5
    
    # Update classifications
    above_function[overlap_mask] = random_assignment
    below_function[overlap_mask] = ~random_assignment
    
    return above_function, below_function

def create_3d_visualization(x, y, z):
    """Create interactive 3D visualization"""
    # Classify points with overlap
    above, below = classify_points(x, y, z, overlap=0.20)
    
    # Create figure
    fig = go.Figure()
    
    # Add points above function (blue)
    fig.add_trace(go.Scatter3d(
        x=x[above],
        y=y[above],
        z=z[above],
        mode='markers',
        marker=dict(
            size=3,
            color='blue',
            opacity=0.6
        ),
        name='Above z=x+y'
    ))
    
    # Add points below function (red)
    fig.add_trace(go.Scatter3d(
        x=x[below],
        y=y[below],
        z=z[below],
        mode='markers',
        marker=dict(
            size=3,
            color='red',
            opacity=0.6
        ),
        name='Below z=x+y'
    ))
    
    # Add the function surface
    x_surf = y_surf = np.linspace(-5, 5, 100)
    X, Y = np.meshgrid(x_surf, y_surf)
    Z = X + Y
    
    fig.add_trace(go.Surface(
        x=X,
        y=Y,
        z=Z,
        opacity=0.3,
        name='z=x+y',
        showscale=False
    ))
    
    # Update layout
    fig.update_layout(
        title='3D Point Distribution Around z=x+y',
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        width=1000,
        height=1000,
        showlegend=True
    )
    
    return fig

if __name__ == "__main__":
    # Generate points
    x, y, z = generate_3d_points(10000)
    
    # Create and save visualization
    fig = create_3d_visualization(x, y, z)
    fig.write_html("function_visualization.html")
    print("Visualization saved to function_visualization.html") 