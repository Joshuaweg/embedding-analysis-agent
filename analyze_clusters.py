import json
import time
from token_table import lookup_tokens
from io import StringIO
import sys
import os

def get_cube_number(node_id):
    # Extract cube number from node_id format "cubeX_clusterY"
    try:
        cube = node_id.split('_')[0]
        return int(cube.replace('cube', ''))
    except:
        return 0

def capture_lookup_output(token_ids):
    # Temporarily redirect stdout to capture lookup_tokens output
    old_stdout = sys.stdout
    result = StringIO()
    sys.stdout = result
    
    # Run lookup_tokens
    lookup_tokens(token_ids)
    
    # Restore stdout and return captured output
    sys.stdout = old_stdout
    return result.getvalue()

def analyze_node_clusters():
    # Create cluster_data directory if it doesn't exist
    if not os.path.exists('cluster_data'):
        os.makedirs('cluster_data')

    # Load the node clusters
    with open('node_clusters.json', 'r', encoding='utf-8') as f:
        graph_data = json.load(f)
    
    nodes = graph_data["nodes"]
    
    # Group nodes by cube number
    cube_groups = {}
    for node_id, node_info in nodes.items():
        cube_num = get_cube_number(node_id)
        if cube_num not in cube_groups:
            cube_groups[cube_num] = []
        cube_groups[cube_num].append((node_id, node_info))
    
    # Process cubes in groups of 100
    sorted_cube_numbers = sorted(cube_groups.keys())
    for group_start in range(0, len(sorted_cube_numbers), 100):
        group_end = min(group_start + 100, len(sorted_cube_numbers))
        current_cube_numbers = sorted_cube_numbers[group_start:group_end]
        
        # Create filename for this group of cubes
        filename = f'cluster_data/cubes_{group_start}_{group_end-1}.txt'
        
        with open(filename, 'w', encoding='utf-8') as outfile:
            outfile.write(f"GPT2 Token Cluster Analysis - Hypercubes {group_start} to {group_end-1}\n")
            outfile.write("=" * 50 + "\n\n")
            
            for cube_num in current_cube_numbers:
                outfile.write(f"\nHypercube {cube_num}\n")
                outfile.write("=" * 50 + "\n")
                
                # Sort clusters within the cube
                cube_nodes = sorted(cube_groups[cube_num], 
                                 key=lambda x: int(x[0].split('_')[1].replace('cluster', '')))
                
                for node_id, node_info in cube_nodes:
                    # Basic node information
                    outfile.write(f"\nCluster: {node_id}\n")
                    outfile.write(f"Size: {node_info['size']}\n")
                    outfile.write(f"Connected to: {', '.join(node_info['connected_nodes'])}\n")
                    
                    # Position
                    pos = node_info['position']
                    outfile.write(f"Position: (x={pos['x']:.3f}, y={pos['y']:.3f})\n")
                    
                    # Tokens
                    outfile.write("\nTokens:\n")
                    outfile.write("-" * 30 + "\n")
                    token_ids = ' '.join(str(tid) for tid in node_info['token_ids'])
                    token_output = capture_lookup_output(token_ids)
                    outfile.write(token_output)
                    outfile.write("\n" + "-"*50 + "\n")
            
            # Print progress
            print(f"Processed hypercubes {group_start} to {group_end-1}")

if __name__ == "__main__":
    print("Starting cluster analysis...")
    analyze_node_clusters()
    print("Analysis complete! Check the cluster_data folder for results.") 