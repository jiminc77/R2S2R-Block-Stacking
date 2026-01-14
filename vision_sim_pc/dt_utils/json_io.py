
import json
import os
import numpy as np

def convert_to_serializable(obj):
    """
    Recursively convert numpy types to standard python types for JSON serialization.
    """
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    return obj

def save_json(filepath, data, indent=4):
    """
    Save data to a JSON file, handling numpy types.
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    serializable_data = convert_to_serializable(data)
    
    with open(filepath, 'w') as f:
        json.dump(serializable_data, f, indent=indent)
    print(f"[INFO] Saved JSON to {filepath}")

def load_json(filepath):
    """
    Load data from a JSON file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
        
    with open(filepath, 'r') as f:
        data = json.load(f)
    print(f"[INFO] Loaded JSON from {filepath}")
    return data
