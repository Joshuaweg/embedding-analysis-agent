from dash import Dash, html, dcc, Input, Output, State, callback_context
import plotly.graph_objects as go
import networkx as nx
from graph_structure import TokenGraph
import json
import numpy as np

class GraphVisualizer:
    def __init__(self):
        self.app = Dash(__name__)
        self.graph = TokenGraph.from_json("node_clusters_2.json")
        self.setup_layout()
        self.setup_callbacks()
        
    def create_network_graph(self, highlight_nodes=None, highlight_path=None):
        """Create network graph visualization"""
        # Create networkx graph for layout
        G = nx.Graph()
        
        # First add all nodes and edges to NetworkX graph
        for node_id, node in self.graph.nodes.items():
            G.add_node(node_id)
            for neighbor in node.connected_nodes:
                G.add_edge(node_id, neighbor)
        
        # Generate layout using NetworkX's spring layout
        pos = nx.spring_layout(
            G,
            k=1/np.sqrt(len(G.nodes())),  # Optimal distance between nodes
            iterations=50,                 # More iterations for better layout
            seed=42                       # For reproducibility
        )
        
        node_sizes = []
        node_texts = []
        node_colors = []
        
        # Process nodes
        for node_id, node in self.graph.nodes.items():
            # Scale node sizes more reasonably
            node_sizes.append(np.sqrt(node.size) * 10)  # Adjusted scaling
            
            # Set node colors
            if highlight_nodes and node_id in highlight_nodes:
                node_colors.append('#ff4444')  # Red for search results
            elif highlight_path and node_id in highlight_path:
                node_colors.append('#ffa500')  # Orange for path
            else:
                node_colors.append('#66b3ff')  # Default blue
            
            # Create hover text
            text = f"Node: {node_id}<br>"
            text += f"Size: {node.size}<br>"
            text += f"Connected to: {', '.join(node.connected_nodes)}<br>"
            text += f"Sample tokens: {', '.join(node.tokens[:5])}"
            node_texts.append(text)
        
        # Create edge traces
        edge_x = []
        edge_y = []
        highlight_edge_x = []
        highlight_edge_y = []
        
        # Process edges
        for node_id, node in self.graph.nodes.items():
            x0, y0 = pos[node_id]
            for neighbor in node.connected_nodes:
                x1, y1 = pos[neighbor]
                
                # Check if edge should be highlighted
                is_highlighted = False
                if highlight_path and len(highlight_path) > 1:
                    for i in range(len(highlight_path) - 1):
                        if (node_id == highlight_path[i] and neighbor == highlight_path[i + 1]) or \
                           (neighbor == highlight_path[i] and node_id == highlight_path[i + 1]):
                            is_highlighted = True
                            break
                
                if is_highlighted:
                    highlight_edge_x.extend([x0, x1, None])
                    highlight_edge_y.extend([y0, y1, None])
                else:
                    edge_x.extend([x0, x1, None])
                    edge_y.extend([y0, y1, None])
        
        # Create figure
        fig = go.Figure()
        
        # Add normal edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#999999'),  # Thinner, lighter edges
            hoverinfo='none',
            mode='lines'
        ))
        
        # Add highlighted edges
        if highlight_edge_x:
            fig.add_trace(go.Scatter(
                x=highlight_edge_x, y=highlight_edge_y,
                line=dict(width=2, color='#ffa500'),  # Thicker highlighted edges
                hoverinfo='none',
                mode='lines'
            ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=[pos[node][0] for node in G.nodes()],
            y=[pos[node][1] for node in G.nodes()],
            mode='markers',
            hoverinfo='text',
            text=node_texts,
            marker=dict(
                size=node_sizes,
                color=node_colors,
                line=dict(width=1, color='#ffffff'),
                opacity=0.8
            )
        ))
        
        # Update layout
        fig.update_layout(
            title="Token Embedding Graph",
            showlegend=False,
            hovermode='closest',
            margin=dict(b=20,l=5,r=5,t=40),
            plot_bgcolor='#111111',
            paper_bgcolor='#111111',
            width=1200,   # Fixed width
            height=800,   # Fixed height
            xaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='#333333',
                zeroline=False,
                showticklabels=False,
                range=[-1.5, 1.5]  # Adjust range to prevent cutoff
            ),
            yaxis=dict(
                showgrid=True,
                gridwidth=1,
                gridcolor='#333333',
                zeroline=False,
                showticklabels=False,
                range=[-1.5, 1.5]  # Adjust range to prevent cutoff
            )
        )
        
        return fig
    
    def setup_layout(self):
        """Create the Dash layout"""
        self.app.layout = html.Div([
            html.H1("Token Embedding Graph"),
            
            # Controls
            html.Div([
                # Search
                html.Div([
                    html.H3("Search Tokens"),
                    dcc.Input(id='search-input', type='text', placeholder='Enter token'),
                    html.Button('Search', id='search-button'),
                ], style={'width': '30%', 'display': 'inline-block'}),
                
                # Path finding
                html.Div([
                    html.H3("Find Path"),
                    dcc.Input(id='start-node', type='text', placeholder='Start node'),
                    dcc.Input(id='end-node', type='text', placeholder='End node'),
                    html.Button('Find Path', id='path-button'),
                ], style={'width': '30%', 'display': 'inline-block'}),
                
                # Random walk
                html.Div([
                    html.H3("Random Walk"),
                    dcc.Input(id='walk-steps', type='number', value=5),
                    html.Button('Start Walk', id='walk-button'),
                ], style={'width': '30%', 'display': 'inline-block'}),
            ]),
            
            # Graph
            dcc.Graph(
                id='token-graph',
                figure=self.create_network_graph(),
                style={'height': '800px'}
            ),
            
            # Info display
            html.Div(id='info-display')
        ])
    
    def setup_callbacks(self):
        """Set up the Dash callbacks"""
        @self.app.callback(
            [Output('token-graph', 'figure'),
             Output('info-display', 'children')],
            [Input('search-button', 'n_clicks'),
             Input('path-button', 'n_clicks'),
             Input('walk-button', 'n_clicks')],
            [State('search-input', 'value'),
             State('start-node', 'value'),
             State('end-node', 'value'),
             State('walk-steps', 'value')]
        )
        def update_graph(search_clicks, path_clicks, walk_clicks,
                        search_term, start_node, end_node, walk_steps):
            ctx = callback_context
            if not ctx.triggered:
                return self.create_network_graph(), "Select an action above"
            
            button_id = ctx.triggered[0]['prop_id'].split('.')[0]
            
            if button_id == 'search-button' and search_term:
                # Search for nodes containing the term
                nodes = self.graph.find_nodes_with_token(search_term)
                node_ids = [n.id for n in nodes]
                return self.create_network_graph(highlight_nodes=node_ids), \
                       f"Found {len(nodes)} nodes containing '{search_term}'"
            
            elif button_id == 'path-button' and start_node and end_node:
                # Find path between nodes
                path = self.graph.bfs_path(start_node, end_node)
                if not path:
                    return self.create_network_graph(), "No path found"
                path_ids = [n.id for n in path]
                return self.create_network_graph(highlight_path=path_ids), \
                       self.graph.get_path_info(path)
            
            elif button_id == 'walk-button':
                # Perform random walk
                walk = self.graph.random_walk(num_steps=walk_steps)
                walk_ids = [n.id for n in walk]
                return self.create_network_graph(highlight_path=walk_ids), \
                       self.graph.get_walk_info(walk)
            
            return self.create_network_graph(), "No action selected"
    
    def run(self, debug=True):
        """Run the Dash app"""
        self.app.run_server(debug=debug)

if __name__ == '__main__':
    visualizer = GraphVisualizer()
    visualizer.run()
