import json
import os
from typing import Dict, List, Any

def fix_json_file():
    # Read the original file
    with open('node_clusters_with_weights.json', 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Initialize the new structure
    fixed_data = {
        "nodes": {}
    }

    # Process each node
    for node_id, node_info in data.items():
        # Ensure node_info is a dictionary
        if isinstance(node_info, list):
            node_info = {"tokens": node_info}

        # Create default values if missing
        fixed_node = {
            "id": node_id,
            "token_ids": list(range(len(node_info.get("tokens", [])))),  # Generate sequential IDs
            "tokens": node_info.get("tokens", []),
            "size": len(node_info.get("tokens", [])),
            "position": node_info.get("position", {"x": 0.0, "y": 0.0}),
            "cluster_spread": node_info.get("cluster_spread", {"std_x": 0.1, "std_y": 0.1}),
            "connected_nodes": node_info.get("connected_nodes", [])
        }

        # Add to fixed data
        fixed_data["nodes"][node_id] = fixed_node

    # Save the fixed file
    backup_file = 'node_clusters_with_weights.json.bak'
    if os.path.exists('node_clusters_with_weights.json'):
        os.rename('node_clusters_with_weights.json', backup_file)

    with open('node_clusters_with_weights_temp.json', 'w', encoding='utf-8') as f:
        json.dump(fixed_data, f, indent=2)

    print(f"Fixed JSON file saved. Original backed up to {backup_file}")

if __name__ == "__main__":
    fix_json_file() 