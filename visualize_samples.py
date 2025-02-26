import numpy as np
import plotly.graph_objects as go
from transformers import GPT2Tokenizer, GPT2Model
from umap import UMAP
import json
from dash import Dash, dcc, html
from dash.dependencies import Input, Output
import networkx as nx

def load_data():
    """Load all required data"""
    # Load the sample indices and coordinates from JSON
    print("Loading sample coordinates...")
    with open('sample_indices_coords.json', 'r') as f:
        sample_data = json.load(f)
        # Convert keys to list and sort them to maintain consistent order
        sample_indices = list(map(int, sample_data.keys()))  # Convert string keys to integers
        sample_indices.sort()  # Sort indices for consistency
        # Get coordinates in the same order as indices
        coords_3d = np.array([sample_data[str(idx)] for idx in sample_indices])
    
    print(f"Loaded {len(sample_indices)} samples")
    print("Coordinates shape:", coords_3d.shape)
    
    # Load persistence features
    print("Loading persistence features...")
    with open('persistence_features.json', 'r') as f:
        features = json.load(f)
    
    # Filter features based on persistence threshold
    PERSISTENCE_THRESHOLD = 0.005
    filtered_features = {
        'h1_features': [
            feature for feature in features['h1_features']
            if feature['death'] - feature['birth'] >= PERSISTENCE_THRESHOLD
        ],
        'h2_features': [
            feature for feature in features['h2_features']
            if feature['death'] - feature['birth'] >= PERSISTENCE_THRESHOLD
        ]
    }
    
    print(f"Filtered features: H1: {len(filtered_features['h1_features'])} (from {len(features['h1_features'])})")
    print(f"Filtered features: H2: {len(filtered_features['h2_features'])} (from {len(features['h2_features'])})")
    
    # Also load persistence results to get cocycle information
    print("Loading persistence results...")
    with open('persistence_results.json', 'r') as f:
        persistence_results = json.load(f)
    
    return np.array(sample_indices), coords_3d, filtered_features, persistence_results

def create_loop_from_points(feature_coords):
    """Create a loop by connecting nearest neighbors"""
    n_points = len(feature_coords)
    if n_points < 3:
        return []
        
    # Start with the first point
    used_points = {0}
    loop = [0]
    
    while len(loop) < n_points:
        current = loop[-1]
        # Find nearest unused point
        distances = [np.linalg.norm(feature_coords[current] - feature_coords[i]) 
                    if i not in used_points else np.inf 
                    for i in range(n_points)]
        next_point = np.argmin(distances)
        
        if distances[next_point] == np.inf:
            break  # No more unused points
            
        loop.append(next_point)
        used_points.add(next_point)
    
    # Close the loop
    loop.append(loop[0])
    return loop

def create_visualization_app():
    """Create Dash app for interactive visualization"""
    app = Dash(__name__)
    
    # Load data
    sample_indices, coords_3d, features, persistence_results = load_data()
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    
    # Create base figure
    base_scatter = go.Scatter3d(
        x=coords_3d[:, 0],
        y=coords_3d[:, 1],
        z=coords_3d[:, 2],
        mode='markers',
        marker=dict(
            size=3,
            color='gray',
            opacity=0.5
        ),
        name='All Points',
        text=[tokenizer.decode([idx]) for idx in sample_indices],
        hoverinfo='text'
    )

    # App layout
    app.layout = html.Div([
        html.H1("Token Embedding Topology Visualization"),
        
        # Controls
        html.Div([
            html.Label('Persistence Threshold:'),
            dcc.Slider(
                id='persistence-slider',
                min=0.01,
                max=0.06,
                step=0.0005,
                value=0.01,
                marks={i/100: f'{i/100:.2f}' for i in range(1, 7, 1)}
            ),
            
            # Feature type selector
            dcc.Checklist(
                id='feature-type',
                options=[
                    {'label': 'H1 Features (Loops)', 'value': 'h1'},
                    {'label': 'H2 Features (Voids)', 'value': 'h2'}
                ],
                value=['h1', 'h2']
            ),
        ], style={'width': '50%', 'padding': '20px'}),
        
        # Graph
        dcc.Graph(
            id='topology-graph',
            style={'height': '800px'}
        ),
        
        # Feature info
        html.Div(id='feature-info')
    ])

    @app.callback(
        [Output('topology-graph', 'figure'),
         Output('feature-info', 'children')],
        [Input('persistence-slider', 'value'),
         Input('feature-type', 'value')]
    )
    def update_graph(threshold, feature_types):
        # Start with just the base scatter
        fig = go.Figure(data=[base_scatter])
        
        active_features = []
        
        # First, identify which features are active based on threshold
        active_h1_features = []
        active_h2_features = []
        
        if 'h1' in feature_types:
            active_h1_features = [(idx, feature) for idx, feature in enumerate(features['h1_features'])
                                if feature['birth'] <= threshold <= feature['death']]
        
        if 'h2' in feature_types:
            active_h2_features = [(idx, feature) for idx, feature in enumerate(features['h2_features'])
                                if feature['birth'] <= threshold <= feature['death']]
        
        # Add H1 features (loops)
        for idx, feature in active_h1_features:
            # Get coordinates for feature points
            feature_coords = coords_3d[[sample_indices.tolist().index(idx) 
                                      for idx in feature['token_indices']]]
            
            # Create loop by connecting nearest neighbors
            loop = create_loop_from_points(feature_coords)
            if not loop:
                continue
            
            # Create a unique group ID for this feature
            group_id = f'H1_{idx}'
            
            # Add points with unique name for this feature
            fig.add_trace(go.Scatter3d(
                x=feature_coords[:, 0],
                y=feature_coords[:, 1],
                z=feature_coords[:, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color='orange',
                    opacity=0.8
                ),
                name=f'H1 Feature {idx} (birth={feature["birth"]:.3f})',
                text=[tokenizer.decode([idx]) for idx in feature['token_indices']],
                hoverinfo='text',
                showlegend=True,
                legendgroup=group_id,
                visible=True
            ))
            
            # Add edges following the loop order
            for i in range(len(loop)-1):
                start, end = loop[i], loop[i+1]
                edge_coords = np.vstack([feature_coords[start], feature_coords[end]])
                fig.add_trace(go.Scatter3d(
                    x=edge_coords[:, 0],
                    y=edge_coords[:, 1],
                    z=edge_coords[:, 2],
                    mode='lines',
                    line=dict(color='orange', width=2),
                    name=f'H1 Feature {idx} Edge',
                    showlegend=False,
                    hoverinfo='none',
                    legendgroup=group_id,
                    visible=True
                ))
            
            active_features.append(('H1', feature))
        
        # Add H2 features (voids)
        for idx, feature in active_h2_features:
            # Get coordinates for feature points
            feature_coords = coords_3d[[sample_indices.tolist().index(idx) 
                                      for idx in feature['token_indices']]]
            
            # Create a unique group ID for this feature
            group_id = f'H2_{idx}'
            
            # Add points with unique name
            fig.add_trace(go.Scatter3d(
                x=feature_coords[:, 0],
                y=feature_coords[:, 1],
                z=feature_coords[:, 2],
                mode='markers',
                marker=dict(
                    size=5,
                    color='green',
                    opacity=0.8
                ),
                name=f'H2 Feature {idx} (birth={feature["birth"]:.3f})',
                text=[tokenizer.decode([idx]) for idx in feature['token_indices']],
                hoverinfo='text',
                showlegend=True,
                legendgroup=group_id,
                visible=True
            ))
            
            # Try to add triangulation
            try:
                from scipy.spatial import ConvexHull
                hull = ConvexHull(feature_coords)
                
                # Add transparent surface with unique name
                fig.add_trace(go.Mesh3d(
                    x=feature_coords[:, 0],
                    y=feature_coords[:, 1],
                    z=feature_coords[:, 2],
                    i=hull.simplices[:, 0],
                    j=hull.simplices[:, 1],
                    k=hull.simplices[:, 2],
                    opacity=0.2,
                    color='green',
                    name=f'H2 Surface {idx}',
                    showlegend=False,
                    hoverinfo='none',
                    legendgroup=group_id,
                    visible=True
                ))
            except:
                print(f"Warning: Could not create surface for H2 feature {idx}")
            
            active_features.append(('H2', feature))
        
        # Update layout
        fig.update_layout(
            title=f'Token Embeddings with Topological Features (threshold={threshold:.3f})',
            scene=dict(
                xaxis_title='UMAP 1',
                yaxis_title='UMAP 2',
                zaxis_title='UMAP 3'
            ),
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01,
                itemclick='toggle',
                itemdoubleclick='toggleothers'
            )
        )
        
        # Create feature info text
        info_text = [
            html.H3(f'Active Features at threshold {threshold:.3f}:'),
            html.Ul([
                html.Li([
                    f'{ftype} Feature:',
                    html.Br(),
                    f'Birth: {feature["birth"]:.3f}',
                    html.Br(),
                    f'Death: {feature["death"]:.3f}',
                    html.Br(),
                    f'Persistence: {feature["persistence"]:.3f}',
                    html.Br(),
                    'Tokens: ' + ', '.join([tokenizer.decode([idx]) for idx in feature['token_indices']])
                ]) for ftype, feature in active_features
            ])
        ]
        
        return fig, info_text

    return app

if __name__ == "__main__":
    app = create_visualization_app()
    app.run_server(debug=True) 